#include<iostream>
using namespace std;

template<size_t NUM_THREADS>
__global__ void batched_sequential_address(
  float* __restrict__ Y,
  float* const __restrict__ X
) {
  __shared__ float shared_data[NUM_THREADS];
  X += blockDim.x * blockIdx.x;

  shared_data[threadIdx.x] = X[threadIdx.x];
  __syncthreads();

  for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
    if(threadIdx.x < stride) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = shared_data[0];
  }
}