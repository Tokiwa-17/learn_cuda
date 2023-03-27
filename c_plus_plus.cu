#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>

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
};

__global__ void kernel(int *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

bool test(std::vector<int, CudaAllocator<int>> &ptr, int n) {
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
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);

    kernel<<<32, 128>>>(arr.data(), n);

    checkCudaErrors(cudaDeviceSynchronize());
    
    if (test(arr, n)) {
        printf("cong.\n");
    } else {
        printf("failed.\n");
    }
    return 0;
}