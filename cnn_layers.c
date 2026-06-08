#include <stdlib.h>
#include <float.h>
#include "cnn_layers.h"

Matrix allocate_matrix(int size) {
    Matrix m;
    m.size = size;
    m.data = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        m.data[i] = (double*)calloc(size, sizeof(double));
    }
    return m;
}

void free_matrix(Matrix* m) {
    for (int i = 0; i < m->size; i++) free(m->data[i]);
    free(m->data);
}

Matrix apply_convolution(Matrix* input, Matrix* filter) {
    int out_size = input->size - filter->size + 1;
    Matrix output = allocate_matrix(out_size);

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            double sum = 0.0;
            for (int fi = 0; fi < filter->size; fi++) {
                for (int fj = 0; fj < filter->size; fj++) {
                    sum += input->data[i + fi][j + fj] * filter->data[fi][fj];
                }
            }
            output.data[i][j] = sum;
        }
    }
    return output;
}

Matrix apply_max_pooling(Matrix* input) {
    int out_size = input->size / 2;
    Matrix output = allocate_matrix(out_size);

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            double max_val = -DBL_MAX;
            for (int pi = 0; pi < 2; pi++) {
                for (int pj = 0; pj < 2; pj++) {
                    double val = input->data[i * 2 + pi][j * 2 + pj];
                    if (val > max_val) max_val = val;
                }
            }
            output.data[i][j] = max_val;
        }
    }
    return output;
}
