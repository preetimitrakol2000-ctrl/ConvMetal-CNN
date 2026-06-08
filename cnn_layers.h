#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

typedef struct {
    int size;
    double** data;
} Matrix;

Matrix allocate_matrix(int size);
void free_matrix(Matrix* m);
Matrix apply_convolution(Matrix* input, Matrix* filter);
Matrix apply_max_pooling(Matrix* input);

#endif
