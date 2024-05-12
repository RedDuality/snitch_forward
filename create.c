
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM 16
#define HIDDEN_DIM 1
#define VOCAB_SIZE 20
// #define N_LAYERS 1
#define N_HEADS 1

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
    // int n_layers = N_LAYERS;
    int vocab_size = VOCAB_SIZE;
    int n_heads = N_HEADS;
    // File name
    char filename[] = "data.h";

    // Open file in write mode ("w" mode)
    file_pointer = fopen(filename, "w");

    int wcls_length = vocab_size * dim;
    int wq_length = dim * dim;
    int wk_length = dim * dim;
    int wv_length = dim * dim;
    int wo_length = dim * dim;

    int w_length = dim * hidden_dim;

    float token_embedding_table[dim];
    float rms_att_weight[dim];
    float rms_ffn_weight[dim];

    float wq[wq_length];
    float wk[wk_length];
    float wv[wv_length];
    float wo[wo_length];

    float w1[w_length];
    float w2[w_length];
    float w3[w_length];

    float rms_final_weight[dim];
    float wcls[wcls_length];

    createValues(dim, token_embedding_table);
    createValues(dim, rms_att_weight);
    createValues(dim, rms_ffn_weight);
    createValues(dim, rms_final_weight);
    createValues(wcls_length, wcls);

    createValues(wq_length, wq);
    createValues(wk_length, wk);
    createValues(wv_length, wv);
    createValues(wo_length, wo);

    createValues(w_length, w1);
    createValues(w_length, w2);
    createValues(w_length, w3);

    fprintf(file_pointer, "float token_embedding_table[%d] = {", dim);
    writeValues(dim, file_pointer, token_embedding_table);
    fprintf(file_pointer, "float rms_att_weight[%d] = {", dim);
    writeValues(dim, file_pointer, rms_att_weight);
    fprintf(file_pointer, "float rms_ffn_weight[%d] = {", dim);
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
    fprintf(file_pointer, "float att[%d];\n", n_heads);
    fprintf(file_pointer, "float logits[%d];\n", vocab_size);
    fprintf(file_pointer, "float key_cache[%d];\n", dim);
    fprintf(file_pointer, "float value_cache[%d];\n", dim);

    fprintf(file_pointer, "\nconst int dim = %d;\n", dim);
    fprintf(file_pointer, "const int hidden_dim = %d;\n", hidden_dim);
    fprintf(file_pointer, "const float vocab_size = %d;\n", vocab_size);
    fprintf(file_pointer, "const float n_heads = %d;", n_heads);

    fclose(file_pointer);

    printf("File created and content written successfully.\n");

    return 0;  // Exit program successfully
}