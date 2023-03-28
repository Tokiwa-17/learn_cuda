#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <type_traits>
#include <iostream>
#include "ticktock.h"
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);
    thrust::device_vector<float> x_device(n);
    thrust::device_vector<float> y_device(n);
    float a = 3.14f;
    auto float_rand = [] {
        return std::rand() * (1.0f / RAND_MAX);
    };
    thrust::generate(x_host.begin(), x_host.end(), float_rand);
    thrust::generate(y_host.begin(), y_host.end(), float_rand);
    x_device = x_host;
    y_device = y_host;
    // for (int i = 0; i < n; i++) {
    //     x[i] = std::rand() + 1.0f / RAND_MAX;
    //     y[i] = std::rand() + 1.0f / RAND_MAX;
    // }
    parallel_for<<<n / 512, 128>>> (n, [a, x = x_device.data(), y = y_device.data()] __device__ (int i) {
        x[i] = a * x[i] + y[i];
    });
    return 0;
}