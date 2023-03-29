#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <type_traits>
#include <iostream>
#include "ticktock.h"

#define EPS 1e-6


template<int N, class T>
__global__ void parallel_sum(T *sum, T *arr) {
    /**
    \brief compute sum of all threads given a block, avoiding data race by allocating a shared memory local_sum. 
    */
    __shared__ volatile T local_sum[1024];
    int j = threadIdx.x, i = blockIdx.x;
    int idx = j + i * blockDim.x;
    local_sum[j] = arr[idx];
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];

    }
    if (j == 0) {
        sum[i] += (local_sum[0] + local_sum[1]);
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
    const int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024);
    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    TICK(shared_memory_sum);
    parallel_sum<n><<<n / 1024, 1024>>>(sum.data(), arr.data());
    TOCK(shared_memory_sum);
    checkCudaErrors(cudaDeviceSynchronize());
    int res = 0;
    for (int i = 0; i < sum.size(); i++) {
        res += sum[i];
    }
    checkCudaErrors(cudaDeviceSynchronize());
    printf("res: %d\n", res);
    if (res == 65536) printf("cong.\n");
    else printf("failed.\n");
    return 0;
}