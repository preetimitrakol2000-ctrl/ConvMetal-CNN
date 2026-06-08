#include <stdio.h>
#include "cnn_layers.h"

int main() {
    printf("🖼️ Initializing ConvMetal-CNN Feature Extraction Engine...\n\n");

    // Creating a synthetic 4x4 image tensor input
    Matrix img = allocate_matrix(4);
    img.data[1][1] = 10.0; img.data[1][2] = 10.0;
    img.data[2][1] = 10.0; img.data[2][2] = 10.0;

    // Creating a standard 3x3 Edge-detection matrix filter
    Matrix kernel = allocate_matrix(3);
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) kernel.data[i][j] = -1.0;
    kernel.data[1][1] = 8.0; // Laplace edge highlight filter config

    Matrix conv_features = apply_convolution(&img, &kernel);
    printf("⚡ Post-Convolution Feature Grid Map Node [0][0]: %.2f\n", conv_features.data[0][0]);

    Matrix pooled_features = apply_max_pooling(&conv_features);
    printf("📉 Post-Max Pooling Downsampled Dimension Extracted: %d x %d\n", pooled_features.size, pooled_features.size);

    free_matrix(&img);
    free_matrix(&kernel);
    free_matrix(&conv_features);
    free_matrix(&pooled_features);
    return 0;
}
