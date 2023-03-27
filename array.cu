#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *ptr, int n) {
    /**
    \brief 低效
    */
    for (int i = 0; i < n; i++) {
        ptr[i] = i;
    }
}

__global__ void kernel2(int *ptr, int n) {
    /**
    \brief 线程数不能太多
    */
    if (threadIdx.x < n) {
        ptr[threadIdx.x] = threadIdx.x;
    }
}

__global__ void kernel3(int *ptr, int n) {
    for (int i = threadIdx.x; i < n; i+= blockDim.x) {
        ptr[i] = i;
    }
}

__global__ void kernel4(int *ptr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        ptr[i] = i;
    }
}

bool test(int *ptr, int n) {
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (ptr[i] != i) {
            flag = false;
            break;
        }
    }
    return flag;
}

int main() {
    int *ptr, n = 65535;
    int threadNum = 128, blockNum = (n + threadNum - 1) / threadNum;
    checkCudaErrors(cudaMallocManaged(&ptr, n * sizeof(int)));
    kernel4<<<blockNum, threadNum>>>(ptr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    if (test(ptr, n)) {
        printf("cong!\n");
    } else {
        printf("wrong.\n");
    }
    // checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(ptr);
    return 0;
}