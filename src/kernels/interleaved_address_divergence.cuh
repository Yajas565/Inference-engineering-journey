#include<iostream>
using namespace std;

template<size_t NUM_THREADS>
__global__ void batched_interleaved_address_naive(
  float* __restrict__ Y,
  float* const __restrict__ X
) {
  __shared__ float shared_data[NUM_THREADS];
  X += blockDim.x * blockIdx.x;

  shared_data[threadIdx.x] = X[threadIdx.x];
  __syncthreads();
  
  for (int stride = 1; stride < NUM_THREADS; stride *= 2) {
    size_t index = 2 * stride * threadIdx.x;
    if(index < NUM_THREADS) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = shared_data[0];
  }
}