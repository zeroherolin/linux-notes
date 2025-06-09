#include <stdio.h>

__global__ void hello_gpu(void) {
    printf("GPU: Hello World!\n");
}

int main(void) {
    printf("CPU: Hello World!\n");

    hello_gpu<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // 等待GPU完成

    return 0;
}

