#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

#define EPS 1e-6

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
template<class T>
__global__ void fill_sin(T *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = sinf(i);
    }
}

__global__ void filter_positive(int *counter, float *res, float const *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (arr[i] >= 0) {
        // 这里有什么问题？请改正：10 分
        int loc = *counter;
        atomicAdd(counter, 1);
        res[loc] = n;
    }
}

int main() {
    constexpr int n = 1<<24;
    std::vector<float, CudaAllocator<float>> arr(n);
    std::vector<float, CudaAllocator<float>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1);

    //FIXME:
    fill_sin<<<n / 1024, 1024>>>(arr.data(), n);

    //FIXME:
    filter_positive<<<n / 1024, 1024>>>(counter.data(), res.data(), arr.data(), n);

    //TODO:
    checkCudaErrors(cudaDeviceSynchronize());

    bool flag = true;
    for (int i = 0; i < n; i++) {
        if ((arr[i] - sinf(i)) > EPS) {
            printf("i: %d\n", i);
            flag = false;
            break;
        }
    }
    if (flag) printf("cong.\n");
    else printf("failed.\n");
    if (counter[0] <= n / 50) {
        printf("Result too short! %d <= %d\n", counter[0], n / 50);
        return -1;
    }
    for (int i = 0; i < counter[0]; i++) {
        if (res[i] < 0) {
            printf("Wrong At %d: %f < 0\n", i, res[i]);
            return -1;  
        }
    }

    printf("All Correct!\n");  
    return 0;
}