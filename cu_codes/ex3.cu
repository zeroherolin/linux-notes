#include <cuda_runtime.h>
#include <stdio.h>

// 将SM版本转换为CUDA核心数
int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } SMtoCores;
    
    SMtoCores gpuArchCoresPerSM[] = {
        {0x30, 192},  // Kepler Generation (SM 3.0) GK10x
        {0x32, 192},  // Kepler Generation (SM 3.2) GK10x
        {0x35, 192},  // Kepler Generation (SM 3.5) GK11x
        {0x37, 192},  // Kepler Generation (SM 3.7) GK21x
        {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x
        {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x
        {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x
        {0x60, 64},   // Pascal Generation (SM 6.0) GP100
        {0x61, 128},  // Pascal Generation (SM 6.1) GP10x
        {0x62, 128},  // Pascal Generation (SM 6.2) GP10x
        {0x70, 64},   // Volta Generation (SM 7.0) GV100
        {0x72, 64},   // Volta Generation (SM 7.2) GV10B
        {0x75, 64},   // Turing Generation (SM 7.5) TU10x
        {0x80, 64},   // Ampere Generation (SM 8.0) GA100
        {0x86, 128},  // Ampere Generation (SM 8.6) GA10x
        {0x87, 128},  // Ampere Generation (SM 8.7) GA10x
        {0x89, 128},  // Ada Lovelace (SM 8.9) AD10x
        {0x90, 128},  // Hopper (SM 9.0) GH100
        {-1, -1}
    };
    
    for (int index = 0; gpuArchCoresPerSM[index].SM != -1; index++) {
        if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return gpuArchCoresPerSM[index].Cores;
        }
    }
    printf("Unknown GPU compute capability %d.%d!\n", major, minor);
    return 128; // 默认值
}

int main() {
    // 1. 查询设备数量
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA capable devices found.\n");
        return -1;
    }
    printf("Found %d CUDA devices\n\n", deviceCount);

    // 2. 遍历所有设备并显示属性
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Compute capability: %d.%d\n", 
               deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", 
               (float)deviceProp.totalGlobalMem/(1024*1024*1024));
        printf("  Multiprocessor count: %d\n", 
               deviceProp.multiProcessorCount);
        printf("  Max threads per block: %d\n", 
               deviceProp.maxThreadsPerBlock);
        printf("  Max block dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
    }

    // 3. 选择性能最好的设备
    int bestDevice = 0;
    int maxCores = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        int cores = prop.multiProcessorCount * 
                   _ConvertSMVer2Cores(prop.major, prop.minor);
        if (cores > maxCores) {
            maxCores = cores;
            bestDevice = dev;
        }
    }
    
    // 4. 设置当前设备
    cudaSetDevice(bestDevice);
    cudaDeviceProp bestProp;
    cudaGetDeviceProperties(&bestProp, bestDevice);
    printf("Selected Device %d: \"%s\" with %d CUDA cores\n",
           bestDevice, bestProp.name, maxCores);

    return 0;
}

