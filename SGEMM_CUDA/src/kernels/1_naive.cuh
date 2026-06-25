__global__ void sgemm_naive(int M, int N, int K, const float* A, const float*B, const float* C, float alpha, float beta) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < M && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; i++) {
      temp += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = temp;
  }
}