#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <type_traits>
#include <iostream>
#include "ticktock.h"

#define EPS 1e-6

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

template <class T>
struct CudaAllocator {
    using value_type = T;
    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }

    // template <class ...Args>
    // void construct(T *p, Args &&...args) {
    //     if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
    //         ::new((void *)p) T(std::forward<Args>(args)...);
    // }
};

template <class T, class Func>
bool test(std::vector<T, CudaAllocator<T>> &ptr, Func func, int n) {
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (ptr[i] - func(i) > EPS) {
            printf("i: %d, %f %f\n", i, ptr[i], func(i));
            flag = false;
            break;
        }
    }
    return flag;
}

int main() {
    int n = 65536;
    std::vector<float, CudaAllocator<float>> arr(n);
    TICK(cpu_sinf);
    parallel_for<<<32, 128>>> (n, [arr = arr.data()] __device__ (int i) {
        arr[i] = sinf(i);
    });
    TOCK(cpu_sinf);
    checkCudaErrors(cudaDeviceSynchronize());
    if(test(arr, sinf, n)) printf("cong.\n");
    else printf("failed.\n");
    return 0;
}