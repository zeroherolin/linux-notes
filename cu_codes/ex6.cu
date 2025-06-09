#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1024 * 1024  // 数据大小
#define STREAM_COUNT 4 // 使用的流数量

// 简单的向量加法核函数
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 主机内存分配
    float *h_a, *h_b, *h_c[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaMallocHost(&h_a, N * sizeof(float)); // 固定内存 (pinned memory)
        cudaMallocHost(&h_b, N * sizeof(float));
        cudaMallocHost(&h_c[i], N * sizeof(float));
        
        // 初始化数据
        for (int j = 0; j < N; j++) {
            h_a[j] = rand() / (float)RAND_MAX;
            h_b[j] = rand() / (float)RAND_MAX;
        }
    }

    // 设备内存分配
    float *d_a[STREAM_COUNT], *d_b[STREAM_COUNT], *d_c[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaMalloc(&d_a[i], N * sizeof(float));
        cudaMalloc(&d_b[i], N * sizeof(float));
        cudaMalloc(&d_c[i], N * sizeof(float));
    }

    // 创建CUDA流
    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 计算执行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 使用多个流并行执行任务
    for (int i = 0; i < STREAM_COUNT; i++) {
        // 异步内存拷贝 (主机->设备)
        cudaMemcpyAsync(d_a[i], h_a, N * sizeof(float), 
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b, N * sizeof(float), 
                        cudaMemcpyHostToDevice, streams[i]);
        
        // 在流中启动核函数
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);
        
        // 异步内存拷贝 (设备->主机)
        cudaMemcpyAsync(h_c[i], d_c[i], N * sizeof(float), 
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步所有流
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 检查结果 (示例：检查最后一个流的第一个元素)
    float tolerance = 1e-5;
    float check = h_a[0] + h_b[0];
    if (fabs(h_c[STREAM_COUNT-1][0] - check) > tolerance) {
        printf("Test FAILED: %f != %f\n", h_c[STREAM_COUNT-1][0], check);
    } else {
        printf("Test PASSED\n");
    }

    // 释放资源
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c[i]);
    }

    return 0;
}

