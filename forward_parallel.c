#include <math.h>
#include <stdio.h>
#include <string.h>

#include "data.h"

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

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

void matmul(float* xout, float* x, float* w, int n, int d, int start, int end) {

    for (int i = start; i < end; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void forward(Transformer * transformer, int token, int pos, int coreindex, int totalcores) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    float *x = s->x;
    int dim = p->dim;

    // token embedding already in x
    // final rmsnorm
    if(coreindex == 0)
    rmsnorm(x, x, w->rms_final_weight, dim);


    int chunk_size = dim/totalcores;
    int startindex = coreindex * chunk_size;
    int finalexindex = coreindex == totalcores-1 ? p->vocab_size : startindex + chunk_size;


    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size,startindex,finalexindex);
}

int main(int argc, char *argv[]) {
    Transformer transformer = {
        .config = config,
        .state = state,
        .weights = weights
    };

    int totalcores = 8;
    int i = 0;
    
    for(int i= 0; i< totalcores; i++){
        forward(&transformer, 1, 0, i, totalcores);
    }

    
    for(int i = 0; i<transformer.config.vocab_size; i++){
        printf("logits[%d]: %.20f\n", i, transformer.state.logits[i]);
    }


    return 0;
}
