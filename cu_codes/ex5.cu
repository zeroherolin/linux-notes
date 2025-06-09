#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

// 常量内存声明
__constant__ int const_factor;

// 使用共享内存的归约核函数
__global__ void reduction_kernel(int* input, int* output) {
    __shared__ int s_data[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载数据到共享内存
    s_data[tid] = input[i];
    __syncthreads();
    
    // 归约操作
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    // 结果乘以常量因子
    if (tid == 0) {
        output[blockIdx.x] = s_data[0] * const_factor;
    }
}

int main() {
    int* h_input = new int[N];
    int* h_output = new int[N / BLOCK_SIZE];
    
    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    
    int *d_input, *d_output;
    size_t input_size = N * sizeof(int);
    size_t output_size = (N / BLOCK_SIZE) * sizeof(int);
    
    // 分配设备内存
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // 分配固定内存用于高效传输
    int* h_pinned;
    cudaMallocHost(&h_pinned, output_size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    // 设置常量内存
    int factor = 2;
    cudaMemcpyToSymbol(const_factor, &factor, sizeof(int));
    
    // 启动核函数
    reduction_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output);
    
    // 拷贝结果到固定内存
    cudaMemcpy(h_pinned, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    int sum = 0;
    for (int i = 0; i < N; i++) sum += i;
    int expected = sum * factor;
    
    int block_sum = 0;
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        block_sum += h_pinned[i];
    }
    
    printf("Computed result: %d\n", block_sum);
    printf("Expected result: %d\n", expected);
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_pinned);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}

