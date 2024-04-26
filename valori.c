
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void writeValues(int number, FILE * file_pointer){
    for(int i = 0; i< number; i++){
        float random = (float)rand() / RAND_MAX;
        // Write content into the file
        if(i != number-1)
            fprintf(file_pointer, "%f,", random);
        else      
            fprintf(file_pointer, "%f", random);  
    }
        // Write content into the file
    fprintf(file_pointer, "%s\n", "};");
}
int main()
{
    srand(time(NULL));

    FILE *file_pointer;

    int dim = 150;
    int vocab_size = 1000;


    // File name
    char filename[] = "valori.txt";

    // Open file in write mode ("w" mode)
    file_pointer = fopen(filename, "w");


    
    fprintf(file_pointer, "\nfloat x[%d] = {", dim);
    writeValues(dim, file_pointer);

    fprintf(file_pointer, "\nfloat logits[%d] = {", dim);
    writeValues(dim, file_pointer);

    fprintf(file_pointer, "\nfloat rms_final_weights[%d] = {", dim);
    writeValues(dim, file_pointer);

    int length = vocab_size*dim + dim;
    fprintf(file_pointer, "\nfloat wcls[%d] = {", length);
    writeValues(length, file_pointer);

    fprintf(file_pointer, "\ntypedef struct {\n\tint dim;\n\tint vocab_size;\n} Config; \n\n");
    fprintf(file_pointer, "\nConfig config = {\n\t.dim = %d,\n\t.vocab_size = %d\n};\n\n", dim, vocab_size);

    fprintf(file_pointer, "\ntypedef struct {\n\tfloat *x;\n\tfloat *logits;\n} RunState;\n\n");
    fprintf(file_pointer, "\nRunState state = {\n\t.x = x,\n\t.logits = logits\n};\n\n");
    fprintf(file_pointer, "\ntypedef struct {\n\t// token embedding table\n\tfloat* rms_final_weight;\n\tfloat* wcls;\n} TransformerWeights;\n\n");
    fprintf(file_pointer, "\nTransformerWeights weights = {\n\t.rms_final_weight = rms_final_weights,\n\t.wcls = wcls\n};\n");

    // Close the file
    fclose(file_pointer);

    printf("File created and content written successfully.\n");

    return 0; // Exit program successfully
}