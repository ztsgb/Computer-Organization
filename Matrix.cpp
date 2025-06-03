#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// 初始化矩阵
void init_matrix(vector<double>& mat, int rows, int cols) {
    mt19937 gen(42);
    uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证结果
bool validate(const vector<double>& A, const vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础版本
void matmul_baseline(const vector<double>& A,
                     const vector<double>& B,
                     vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// OpenMP并行版本
void matmul_openmp(const vector<double>& A,
                   const vector<double>& B,
                   vector<double>& C, int N, int M, int P) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

// 子块并行优化
void matmul_block_tiling(const vector<double>& A,
                         const vector<double>& B,
                         vector<double>& C, int N, int M, int P, int block_size = 64) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < P; jj += block_size)
            for (int kk = 0; kk < M; kk += block_size)
                for (int i = ii; i < min(ii + block_size, N); ++i)
                    for (int j = jj; j < min(jj + block_size, P); ++j) {
                        double sum = 0;
                        for (int k = kk; k < min(kk + block_size, M); ++k)
                            sum += A[i * M + k] * B[k * P + j];
                        C[i * P + j] += sum;
                    }
}

// OpenMP+循环展开优化
void matmul_openmp_unroll(const vector<double>& A,
                          const vector<double>& B,
                          vector<double>& C, int N, int M, int P) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            double a = A[i * M + k];
            // 展开4次循环
            int j = 0;
            for (; j + 3 < P; j += 4) {
                C[i * P + j] += a * B[k * P + j];
                C[i * P + j + 1] += a * B[k * P + j + 1];
                C[i * P + j + 2] += a * B[k * P + j + 2];
                C[i * P + j + 3] += a * B[k * P + j + 3];
            }
            // 处理剩余部分
            for (; j < P; ++j) {
                C[i * P + j] += a * B[k * P + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    string mode = argc >= 2 ? argv[1] : "baseline";

    vector<double> A(N * M);
    vector<double> B(M * P);
    vector<double> C(N * P, 0);
    vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);

    // 计算参考结果
    auto start_ref = high_resolution_clock::now();
    matmul_baseline(A, B, C_ref, N, M, P);
    auto end_ref = high_resolution_clock::now();
    auto duration_ref = duration_cast<milliseconds>(end_ref - start_ref);

    if (mode == "baseline") {
        cout << "[Baseline] Time: " << duration_ref.count() << " ms\n";
    } else {
        auto start = high_resolution_clock::now();
        
        if (mode == "openmp") {
            matmul_openmp(A, B, C, N, M, P);
        } else if (mode == "block") {
            matmul_block_tiling(A, B, C, N, M, P);
        } else if (mode == "unroll") {  // 新增循环展开模式
            matmul_openmp_unroll(A, B, C, N, M, P);
        } else {
            cerr << "Usage: ./program [baseline|openmp|block|unroll]" << endl;
            return 1;
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        cout << "[" << mode << "] Time: " << duration.count() << " ms" << endl;
        cout << "[" << mode << "] Speedup: " << (double)duration_ref.count() / duration.count() << endl;
        cout << "[" << mode << "] Valid: " << validate(C, C_ref, N, P) << endl;
    }
    
    return 0;
}