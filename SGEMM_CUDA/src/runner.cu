#include<iostream>
using namespace std;



void run_segmm_naive(int M, int N, int K, const float* A, const float* B, const float* C, float alpha, float beta) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C, alpha, beta);
}

void run_segmm_global_mem_coalesce(int M, int N, int K, const float* A, const float* B, const float* C, float alpha, float beta) {
  dim3 blockDim(32 * 32);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, A, B, C, alpha, beta);
}