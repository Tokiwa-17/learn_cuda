#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *ptr) {
    *ptr = 47;
}

int main() {
    int *ptr;
    checkCudaErrors(cudaMalloc(&ptr, sizeof(int)));
    kernel<<<1, 1>>>(ptr);
    int ret;
    checkCudaErrors(cudaMemcpy(&ret, ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d\n", ret);
    // checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(ptr);
    return 0;
}