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
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);
    // init
    parallel_for<<<n / 512, 128>>> (n, [arr = arr.data(), sum = sum.data()] __device__ (int i) {
        arr[i] = 1;
    });
    checkCudaErrors(cudaDeviceSynchronize());
    if (test(arr, [](int i) {return 1;}, n)) printf("init success.\n");
    else printf("init failed.\n");
    TICK(atomic_op);
    parallel_for<<<n / 512, 128>>> (n, [arr = arr.data(), sum = sum.data()] __device__ (int i) {
        // sum[0] += arr[i]; FIXME: data race!
        atomicAdd(sum, arr[i]);
    });
    TOCK(atomic_op);
    checkCudaErrors(cudaDeviceSynchronize());
    // if(test(arr, sinf, n)) printf("cong.\n");
    // else printf("failed.\n");
    printf("%d\n", sum[0]);
    return 0;
}