
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM 16
#define HIDDEN_DIM 10
#define VOCAB_SIZE 20
#define N_LAYERS 3
#define N_HEADS 8
#define N_KV_HEADS 1
#define SEQ_LEN 1

void createValues(int size, float *array) {
    for (int i = 0; i < size; i++)
        array[i] = (float)rand() / RAND_MAX;
}

void writeValues(int length, FILE *file_pointer, float *array) {
    for (int i = 0; i < length; i++) {
        fprintf(file_pointer, "%f", array[i]);

        if (i != length - 1)
            fprintf(file_pointer, ",");
    }
    fprintf(file_pointer, "};\n");
}

int main() {
    srand(time(NULL));

    FILE *file_pointer;

    int dim = DIM;
    int hidden_dim = HIDDEN_DIM;
    int n_layers = N_LAYERS;
    int vocab_size = VOCAB_SIZE;
    int n_heads = N_HEADS;
    int n_kv_heads = N_KV_HEADS;
    int seq_len = SEQ_LEN;


    int head_size;
    if(dim % n_heads == 0){
        head_size = dim / n_heads;
    } else {
        printf("dim must be a multiple of n_heads!");
        return -1;
    }



    // File name
    char filename[] = "data.h";

    // Open file in write mode ("w" mode)
    file_pointer = fopen(filename, "w");

    int token_embedding_length = vocab_size * dim;
    int att_length = n_layers * dim; 
    int ffn_length = n_layers * dim;
    //note dim == n_heads * head_size
    int wq_length = n_layers * dim * dim;
    int wk_length = n_layers * dim * n_kv_heads * head_size;
    int wv_length = n_layers * dim * n_kv_heads * head_size;
    int wo_length = n_layers * dim * dim;
    
    int w_length = n_layers * dim * hidden_dim;
    int wcls_length = vocab_size * dim;


    float token_embedding_table[token_embedding_length];
    float rms_att_weight[att_length];
    float rms_ffn_weight[ffn_length];

    float wq[wq_length];
    float wk[wk_length];
    float wv[wv_length];
    float wo[wo_length];

    float w1[w_length];
    float w2[w_length];
    float w3[w_length];

    float rms_final_weight[dim];
    float wcls[wcls_length];

    createValues(token_embedding_length, token_embedding_table);
    createValues(att_length, rms_att_weight);
    createValues(ffn_length, rms_ffn_weight);
    createValues(dim, rms_final_weight);
    createValues(wcls_length, wcls);

    createValues(wq_length, wq);
    createValues(wk_length, wk);
    createValues(wv_length, wv);
    createValues(wo_length, wo);

    createValues(w_length, w1);
    createValues(w_length, w2);
    createValues(w_length, w3);

    fprintf(file_pointer, "float token_embedding_table[%d] = {", token_embedding_length);
    writeValues(dim, file_pointer, token_embedding_table);
    fprintf(file_pointer, "float rms_att_weight[%d] = {", att_length);
    writeValues(dim, file_pointer, rms_att_weight);
    fprintf(file_pointer, "float rms_ffn_weight[%d] = {", ffn_length);
    writeValues(dim, file_pointer, rms_ffn_weight);

    fprintf(file_pointer, "float wq[%d] = {", wq_length);
    writeValues(wq_length, file_pointer, wq);
    fprintf(file_pointer, "float wk[%d] = {", wk_length);
    writeValues(wk_length, file_pointer, wk);
    fprintf(file_pointer, "float wv[%d] = {", wv_length);
    writeValues(wv_length, file_pointer, wv);
    fprintf(file_pointer, "float wo[%d] = {", wo_length);
    writeValues(wo_length, file_pointer, wo);

    fprintf(file_pointer, "float w1[%d] = {", w_length);
    writeValues(w_length, file_pointer, w1);
    fprintf(file_pointer, "float w2[%d] = {", w_length);
    writeValues(w_length, file_pointer, w2);
    fprintf(file_pointer, "float w3[%d] = {", w_length);
    writeValues(w_length, file_pointer, w3);

    fprintf(file_pointer, "float rms_final_weight[%d] = {", dim);
    writeValues(dim, file_pointer, rms_final_weight);
    fprintf(file_pointer, "float wcls[%d] = {", wcls_length);
    writeValues(wcls_length, file_pointer, wcls);

    fprintf(file_pointer, "\nfloat x[%d];\n", dim);
    fprintf(file_pointer, "float xb[%d];\n", dim);
    fprintf(file_pointer, "float xb2[%d];\n", dim);
    fprintf(file_pointer, "float hb[%d];\n", hidden_dim);
    fprintf(file_pointer, "float hb2[%d];\n", hidden_dim);
    fprintf(file_pointer, "float q[%d];\n", dim);
    fprintf(file_pointer, "float k[%d];\n", dim);
    fprintf(file_pointer, "float v[%d];\n", dim);
    fprintf(file_pointer, "float att[%d];\n", n_heads * seq_len);
    fprintf(file_pointer, "float logits[%d];\n", vocab_size);
    fprintf(file_pointer, "float key_cache[%d];\n", n_layers * seq_len * dim);
    fprintf(file_pointer, "float value_cache[%d];\n", n_layers * seq_len * dim);

    fprintf(file_pointer, "\nconst int dim = %d;\n", dim);
    fprintf(file_pointer, "const int hidden_dim = %d;\n", hidden_dim);
    fprintf(file_pointer, "const int n_layers = %d;\n", n_layers);
    fprintf(file_pointer, "const float vocab_size = %d;\n", vocab_size);
    fprintf(file_pointer, "const float n_heads = %d;\n", n_heads);
    fprintf(file_pointer, "const float n_kv_heads = %d;\n", n_kv_heads);
    fprintf(file_pointer, "const float seq_len = %d;", seq_len);

    fclose(file_pointer);

    printf("File created and content written successfully.\n");

    return 0;  // Exit program successfully
}