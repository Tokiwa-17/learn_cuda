#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <type_traits>
#include <iostream>
#include "ticktock.h"

#define EPS 1e-6

template<class T>
__global__ void parallel_transpose(T *matrix, T *matrix_trans, const int nx, const int ny) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x = idx % nx, y = idx / nx;
    if (x >= nx || y >= ny) return;
    matrix_trans[y * nx + x] = matrix[x * ny + y];
}

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

template<int nx, int ny, class T> 
void printMatrix(T *mat) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%d ", mat[j + i * ny]);
        }
        printf("\n");
    }
}

int main() {
    const int nx = 1 << 12, ny = 1 << 11;
    int blockSize = 1024, gridSize = nx * ny / blockSize;
    std::vector<int, CudaAllocator<int>>matrix(nx * ny);
    std::vector<int, CudaAllocator<int>>matrix_trans(nx * ny);
    for (int i = 0; i < nx * ny; i++) {
        matrix[i] = i;
    }
    // checkCudaErrors(cudaDeviceSynchronize());
    // printMatrix<nx, ny>(matrix.data());
    TICK(matrix_transpose);
    parallel_transpose<<<gridSize, blockSize>>>(matrix.data(), matrix_trans.data(), nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(matrix_transpose);
    // printMatrix<ny, nx>(matrix_trans.data());
    bool flag = true;
    for (int i = 0; i < nx; i++) { // row
        for (int j = 0; j < ny; j++) { // col
            if (matrix[i * ny + j] != matrix_trans[j * nx + i]) {
                flag = false;
                break;
            }
        }
    }
    if (flag) printf("cong.\n");
    else printf("error.\n");
    return 0;
}