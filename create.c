
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM 16
#define VOCAB_SIZE 20

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
    fprintf(file_pointer, "%s\n", "};");
}

int main() {
    srand(time(NULL));

    FILE *file_pointer;

    int dim = DIM;
    int vocab_size = VOCAB_SIZE;

    // File name
    char filename[] = "data.h";

    // Open file in write mode ("w" mode)
    file_pointer = fopen(filename, "w");

    float x[dim];
    float logits[vocab_size];
    float rms_final_weight[dim];

    int wcls_length = vocab_size * dim + dim;
    float wcls[wcls_length];

    createValues(dim, x);
    createValues(dim, rms_final_weight);
    createValues(wcls_length, wcls);


    fprintf(file_pointer, "float x[%d] = {", dim);
    writeValues(dim, file_pointer, x);

    fprintf(file_pointer, "\nfloat logits[%d];", vocab_size);

    fprintf(file_pointer, "\nfloat rms_final_weight[%d] = {", dim);
    writeValues(dim, file_pointer, rms_final_weight);

    fprintf(file_pointer, "\nfloat wcls[%d] = {", wcls_length);
    writeValues(wcls_length, file_pointer, wcls);

    fprintf(file_pointer, "\ntypedef struct {\n\tint dim;\n\tint vocab_size;\n} Config; \n");
    fprintf(file_pointer, "\nConfig config = {\n\t.dim = %d,\n\t.vocab_size = %d\n};\n", dim, vocab_size);

    fprintf(file_pointer, "\ntypedef struct {\n\tfloat *x;\n\tfloat *logits;\n} RunState;\n");
    fprintf(file_pointer, "\nRunState state = {\n\t.x = x,\n\t.logits = logits\n};\n");
    fprintf(file_pointer, "\ntypedef struct {\n\t// token embedding table\n\tfloat* rms_final_weight;\n\tfloat* wcls;\n} TransformerWeights;\n");
    fprintf(file_pointer, "\nTransformerWeights weights = {\n\t.rms_final_weight = rms_final_weight,\n\t.wcls = wcls\n};");

    fclose(file_pointer);

    printf("File created and content written successfully.\n");

    return 0;  // Exit program successfully
}