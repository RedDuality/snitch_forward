#include "data.h"

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int vocab_size;
    int n_heads;
    int n_kv_heads;
    int seq_len;
} Config;

Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .vocab_size = vocab_size,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .seq_len = seq_len,
};

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

TransformerWeights weights = {
    .token_embedding_table = token_embedding_table,
    .rms_att_weight = rms_att_weight,
    .rms_ffn_weight = rms_ffn_weight,
    .wq = wq,
    .wk = wk,
    .wv = wv,
    .wo = wo,
    .w1 = w1,
    .w2 = w2,
    .w3 = w3,
    .wcls = wcls,
    .rms_final_weight = rms_final_weight,
};

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} RunState;

RunState state = {
    .x = x,
    .xb = xb,
    .xb2 = xb2,
    .hb = hb,
    .hb2 = hb2,
    .q = q,
    .k = k,
    .v = v,
    .att = att,
    .logits = logits,
    .key_cache = key_cache,
    .value_cache = value_cache,
};

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;
