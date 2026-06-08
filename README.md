# ConvMetal-CNN

A hardware-friendly inference engine executing Convolutional Neural Network (CNN) feature extraction layers entirely in pure C. This project bypasses heavy runtimes to perform dynamic spatial matrix pooling and matrix convolutions via deterministic cache allocation.

## ⚡ Architectural Mechanics
* **Matrix Convolutions:** Manual element-wise matrix multi-channel dot products.
* **Max-Pooling Node:** Down-samples matrix representations to isolate core structural features.

## 📦 Compilation & Execution
```bash
gcc main.c cnn_layers.c -o convmetal -lm
./convmetal
