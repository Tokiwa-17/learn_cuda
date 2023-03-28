#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <type_traits>

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

struct MyFunctor {
    __device__ void operator() (int i) const {
        printf("number: %d\n", i);
    }
};

int main() {
    int n = 65536;
    parallel_for<<<32, 128>>> (n, MyFunctor{});
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}