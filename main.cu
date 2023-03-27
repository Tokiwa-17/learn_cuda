#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

__host__ __device__ void say_hello() {
#ifdef __CUDA_ARCH__
    printf("Hello, world from block %d of %d, thread %d of %d, GPU %d!\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x, __CUDA_ARCH__);

    // unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // unsigned int tnum = blockDim.x * gridDim.x;
    
#else
    printf("Hello, world from CPU!\n");
#endif
}

__global__ void kernel() {
    say_hello();
}

constexpr const char *cuthead(const char *p) {
    return p + 1;
}

int main() {
    //kernel<<<2, 3>>>();
    kernel<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();
    cudaDeviceSynchronize();
    say_hello();
    printf(cuthead("Cello, world\n"));
    return 0;
}