#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#define N 1024
#define M 2024
#define P 512
#define BLOCK_SIZE 16

__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
    return;  // 添加return语句
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
    return;
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
    return;
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // HIP implementation
    double *d_A, *d_B, *d_C;
    
    hipMalloc(&d_A, N * M * sizeof(double));
    hipMalloc(&d_B, M * P * sizeof(double));
    hipMalloc(&d_C, N * P * sizeof(double));
    
    hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    auto dcu_start = std::chrono::high_resolution_clock::now();
    hipDeviceSynchronize();
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N, M, P);
    hipDeviceSynchronize();
    auto dcu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dcu_duration = dcu_end - dcu_start;
    std::cout << "DCU time: " << dcu_duration.count() << " seconds" << std::endl;
    
    double speedup = cpu_duration.count() / dcu_duration.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    hipMemcpy(C.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);
    
    if (validate(C_ref, C)) {
        std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
        std::cout << "[HIP] Valid: 0" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}