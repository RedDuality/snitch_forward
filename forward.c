
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "data.h"

typedef struct {
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass
} Transformer;

int checkValues(float * current, float * correct, int length){

    for (int i = 0; i< length; i++){
        if(current[i] != correct[i])
            return 1;
    }
    printf("uguali!");
    return 0;
}

void writeResults(int number, FILE *file_pointer, float *array)
{
    for (int i = 0; i < number; i++)
    {
        // Write content into the file
        if (i != number - 1)
            fprintf(file_pointer, "%.16f,", array[i]);
        else
            fprintf(file_pointer, "%.16f", array[i]);
    }
    // Write content into the file
    fprintf(file_pointer, "%s\n", "};");
}

void writeOnFile(int length, float * array){
    char filename[] = "results.h";

    // Open file in write mode ("w" mode)
    FILE *file_pointer = fopen(filename, "w");

    fprintf(file_pointer, "\nfloat result[%d] = {", length);
    writeResults(length, file_pointer, array);

    // Close the file
    fclose(file_pointer);

    printf("File created and content written successfully.\n");
}


void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

/*
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
        x[i] = my_expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}*/

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void forward(Transformer * transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    float *x = s->x;
    int dim = p->dim;

    // token embedding already in x
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}


int main(int argc, char *argv[]) {
    Transformer transformer = {
        .config = config,
        .state = state,
        .weights = weights
    };

    forward(&transformer, 1, 0);

    
    for(int i = 0; i<transformer.config.vocab_size; i++){
        printf("logits[%d]: %.20f\n", i, transformer.state.logits[i]);
    }

    writeOnFile(transformer.config.vocab_size, transformer.state.logits);

    //return checkValues(transformer.state.logits, result, transformer.config.vocab_size);
}
