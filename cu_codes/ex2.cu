#include <stdio.h>

__global__ void hello_gpu(void) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    printf("GPU: Hello World! block:%d thread:%d id:%d\n", bid, tid, idx);
}

int main(void) {
    hello_gpu<<<2, 4>>>(); // 启动2个块，每个块4个线程

    cudaDeviceSynchronize();

    return 0;
}

