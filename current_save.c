#include <stdio.h>
#include <string.h>

#include "results.h"
#include "snrt.h"
#include "structs.h"

int checkValues(float* current, float* correct, int length) {
    for (int i = 0; i < length; i++) {
        if (current[i] != correct[i])
            return (int)(current[i] * 10000000);
    }
    return 0;
}

void barrierEater(int barriers) {
    for (int i = 0; i < barriers; i++)
        snrt_cluster_hw_barrier();
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d, int start, int end) {
    
    for (int i = start; i < end; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    int totalcores = snrt_cluster_compute_core_num();
    int coreindex = snrt_cluster_core_idx();

    int mod = dim % totalcores;
    int chunk_size = coreindex < mod ? p->dim / totalcores + 1 : p->dim / totalcores;
    int start = coreindex < mod ? coreindex * chunk_size: coreindex * chunk_size + mod;
    int end = start + chunk_size;

    int kv_mod = kv_dim % totalcores;
    int kv_chunk_size = coreindex < kv_mod ? kv_dim / totalcores + 1 : kv_dim / totalcores;
    int kv_start = coreindex < kv_mod ? coreindex * kv_chunk_size : coreindex * kv_chunk_size + kv_mod;
    int kv_end = kv_start + kv_chunk_size;

    int hd_mod = hidden_dim % totalcores;
    int hd_chunk_size = coreindex < hd_mod ? hidden_dim / totalcores + 1 : hidden_dim / totalcores;
    int hd_start = coreindex < hd_mod ? coreindex * hd_chunk_size : coreindex * hd_chunk_size + hd_mod;
    int hd_end = hd_start + hd_chunk_size;

    int heads_mod = p->n_heads % totalcores;
    int heads_chunk_size = coreindex < heads_mod ? p->n_heads / totalcores + 1 : p->n_heads / totalcores;
    int heads_start = coreindex < heads_mod ? coreindex * heads_chunk_size : coreindex * heads_chunk_size + heads_mod;
    int heads_end = heads_start + heads_chunk_size;

    int vocab_mod = p->vocab_size % totalcores;
    int vocab_chunk_size = coreindex < vocab_mod ? p->vocab_size / totalcores + 1 : p->vocab_size / totalcores;
    int vocab_start = coreindex < vocab_mod ? coreindex * vocab_chunk_size : coreindex * vocab_chunk_size;
    int vocab_end = vocab_start + vocab_chunk_size;

    // copy the token embedding into x
    if (coreindex == 0) {
        float* content_row = w->token_embedding_table + token * dim;
        memcpy(x, content_row, dim * sizeof(*x));
    }

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        if (coreindex == 0)
            // attention rmsnorm
            rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        snrt_cluster_hw_barrier();

        // kv cache layer offset for convenience
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim, start, end);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim, kv_start, kv_end);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim, kv_start, kv_end);
        snrt_cluster_hw_barrier();

        if (coreindex == 0) {
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;  // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }

        
        snrt_cluster_hw_barrier();
        // multihead attention. iterate over all heads
        int h;
        for (h = heads_start; h < heads_end; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        snrt_cluster_hw_barrier();

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim, start, end);
        // residual connection back into x
        for (int i = start; i < end; i++) {
            x[i] += s->xb2[i];
        }
        snrt_cluster_hw_barrier();
        if (coreindex == 0)
            // ffn rmsnorm
            rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        snrt_cluster_hw_barrier();
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, hd_start, hd_end);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim, hd_start, hd_end);

        // SwiGLU non-linearity
        for (int i = hd_start; i < hd_end; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        snrt_cluster_hw_barrier();
        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, start, end);

        // residual connection
        for (int i = start; i < end; i++) {
            x[i] += s->xb[i];
        }
        snrt_cluster_hw_barrier();
    }

    if (coreindex == 0)
        rmsnorm(x, x, w->rms_final_weight, dim);
    snrt_cluster_hw_barrier();

    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size, vocab_start, vocab_end);

    snrt_cluster_hw_barrier();
}

int main(int argc, char* argv[]) {
    Transformer transformer = {
        .config = config,
        .state = state,
        .weights = weights};

    if (snrt_is_compute_core())
        forward(&transformer, 0, 0);
    else
        barrierEater(8 * transformer.config.n_layers + 2);

    if (snrt_is_dm_core())
        return checkValues(transformer.state.logits, result, transformer.config.vocab_size);

    return 0;
}