#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000  // 数据量
#define STREAM_COUNT 4  // 使用的流数量
#define THREADS_PER_BLOCK 256

// CUDA核函数：向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 故意增加计算量以延长执行时间
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            sum += a[idx] + b[idx];
        }
        c[idx] = sum;
    }
}

// 检查CUDA运行时错误
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // 分配主机内存
    float *h_a, *h_b, *h_c[STREAM_COUNT];
    size_t size = N * sizeof(float);
    
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    for (int i = 0; i < STREAM_COUNT; i++) {
        h_c[i] = (float*)malloc(size);
    }

    // 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // 分配设备内存
    float *d_a, *d_b;
    float *d_c[STREAM_COUNT];
    
    checkCudaError(cudaMalloc(&d_a, size), "Allocate d_a");
    checkCudaError(cudaMalloc(&d_b, size), "Allocate d_b");
    for (int i = 0; i < STREAM_COUNT; i++) {
        checkCudaError(cudaMalloc(&d_c[i], size), "Allocate d_c");
    }

    // 创建CUDA流和事件
    cudaStream_t streams[STREAM_COUNT];
    cudaEvent_t startEvents[STREAM_COUNT], stopEvents[STREAM_COUNT];
    cudaEvent_t globalStart, globalStop;
    
    checkCudaError(cudaEventCreate(&globalStart), "Create globalStart event");
    checkCudaError(cudaEventCreate(&globalStop), "Create globalStop event");
    
    for (int i = 0; i < STREAM_COUNT; i++) {
        checkCudaError(cudaStreamCreate(&streams[i]), "Create stream");
        checkCudaError(cudaEventCreate(&startEvents[i]), "Create start event");
        checkCudaError(cudaEventCreate(&stopEvents[i]), "Create stop event");
    }

    // 计算网格大小
    int blockSize = THREADS_PER_BLOCK;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 记录全局开始时间
    checkCudaError(cudaEventRecord(globalStart, 0), "Record global start");
    
    // 异步复制数据到设备并启动核函数
    for (int i = 0; i < STREAM_COUNT; i++) {
        // 计算当前流的数据偏移量
        int offset = i * (N / STREAM_COUNT);
        int streamSize = N / STREAM_COUNT;
        
        // 为当前流复制数据
        checkCudaError(cudaMemcpyAsync(
            d_a + offset, 
            h_a + offset, 
            streamSize * sizeof(float), 
            cudaMemcpyHostToDevice, 
            streams[i]
        ), "Memcpy H2D a");
        
        checkCudaError(cudaMemcpyAsync(
            d_b + offset, 
            h_b + offset, 
            streamSize * sizeof(float), 
            cudaMemcpyHostToDevice, 
            streams[i]
        ), "Memcpy H2D b");
        
        // 记录流开始事件
        checkCudaError(cudaEventRecord(startEvents[i], streams[i]), "Record start event");
        
        // 启动核函数
        vectorAdd<<<gridSize / STREAM_COUNT, blockSize, 0, streams[i]>>>(
            d_a + offset, d_b + offset, d_c[i] + offset, streamSize);
        
        // 记录流结束事件
        checkCudaError(cudaEventRecord(stopEvents[i], streams[i]), "Record stop event");
        
        // 将结果复制回主机
        checkCudaError(cudaMemcpyAsync(
            h_c[i] + offset, 
            d_c[i] + offset,
            streamSize * sizeof(float),
            cudaMemcpyDeviceToHost, 
            streams[i]
        ), "Memcpy D2H");
    }

    // 记录全局结束时间并等待所有操作完成
    checkCudaError(cudaEventRecord(globalStop, 0), "Record global stop");
    checkCudaError(cudaEventSynchronize(globalStop), "Sync global stop");

    // 计算并打印执行时间
    float totalTime;
    checkCudaError(cudaEventElapsedTime(&totalTime, globalStart, globalStop),
                "Calculate global time");
    
    printf("Total execution time: %.2f ms\n", totalTime);
    
    // 打印每个流的执行时间
    for (int i = 0; i < STREAM_COUNT; i++) {
        float streamTime;
        checkCudaError(cudaEventElapsedTime(&streamTime, startEvents[i], stopEvents[i]),
                    "Calculate stream time");
        printf("Stream %d kernel time: %.2f ms\n", i, streamTime);
    }

    // 清理资源
    checkCudaError(cudaEventDestroy(globalStart), "Destroy globalStart");
    checkCudaError(cudaEventDestroy(globalStop), "Destroy globalStop");
    
    for (int i = 0; i < STREAM_COUNT; i++) {
        checkCudaError(cudaStreamDestroy(streams[i]), "Destroy stream");
        checkCudaError(cudaEventDestroy(startEvents[i]), "Destroy start event");
        checkCudaError(cudaEventDestroy(stopEvents[i]), "Destroy stop event");
        free(h_c[i]);
    }
    
    free(h_a);
    free(h_b);
    checkCudaError(cudaFree(d_a), "Free d_a");
    checkCudaError(cudaFree(d_b), "Free d_b");
    for (int i = 0; i < STREAM_COUNT; i++) {
        checkCudaError(cudaFree(d_c[i]), "Free d_c");
    }

    return 0;
}