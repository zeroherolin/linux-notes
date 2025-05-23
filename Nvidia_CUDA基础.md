# CUDA编程基础

## 核函数

- 源码

ex1.cu

```c++
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

    cudaDeviceSynchronize();

    return 0;
}
```

- 执行

```bash
nvcc ex1.cu -o ex1
./ex1
```

- 输出

> CPU: Hello World! \
GPU: Hello World!

## 线程模型

### 多维线程块中的线程索引

```c++
int tid = threadIdx.z * blockDim.x * blockDim.y +
          threadIdx.y * blockDim.x + 
          threadIdx.x;
```

### 多维网格中的线程块索引

```c++
int bid = blockIdx.z * blockDim.x * blockDim.y +
          blockIdx.y * blockDim.x + 
          blockIdx.x;
```

### 多维网格中的线程唯一索引

```c++
int idx = bid * (blockDim.x * blockDim.y * blockDim.z) + tid;
```

- 源码

ex2.cu

```c++
#include <stdio.h>

__global__ void hello_gpu(void) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    printf("GPU: Hello World! block:%d thread:%d id:%d\n", bid, tid, idx);
}

int main(void) {
    hello_gpu<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}
```

- 执行

```bash
nvcc ex2.cu -o ex2
./ex2
```

- 输出

> GPU: Hello World! block:1 thread:0 id:4 \
GPU: Hello World! block:1 thread:1 id:5 \
GPU: Hello World! block:1 thread:2 id:6 \
GPU: Hello World! block:1 thread:3 id:7 \
GPU: Hello World! block:0 thread:0 id:0 \
GPU: Hello World! block:0 thread:1 id:1 \
GPU: Hello World! block:0 thread:2 id:2 \
GPU: Hello World! block:0 thread:3 id:3

## 设置Device

- 源码

ex3.cu

```c++
#include <stdio.h>

int main(void) {
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    } else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    } else {
        printf("set GPU 0 for computing.\n");
    }

    return 0;
}
```

- 执行

```c++
nvcc ex3.cu -o ex3
./ex3
```

- 输出

> The count of GPUs is 1. \
set GPU 0 for computing.

## 内存管理

| 标准C内存管理函数 | CUDA内存管理函数 | 主机代码 | 设备代码 |
| -------------- | -------------- | ------- | ------- |
| malloc  | cudaMalloc | `float *fpHost_A;` `fpHost_A = (float *)malloc(nBytes);` | `float *fpDevice_A;` `cudaMalloc((float**)&fpDevice_A, nBytes);` |
| memcpy  | cudaMemcpy | `memcpy((void*)d, (void*)s, nBytes);`                    | `cudaMemcpy(Device_A, Host_A, nBytes, cudaMemcpyHostToHost);`    |
| memset  | cudaMemset | `memset(fpHost_A, 0, nBytes);`                           | `cudaMemset(fpDevice_A, 0, nBytes);`                             |
| free    | cudaFree   | `free(pHost_A);`                                         | `cudaFree(pDevice_A);`                                           |

***
👉 Updating...
⭐ I like your Star!
🔙 [Go Back](README.md)