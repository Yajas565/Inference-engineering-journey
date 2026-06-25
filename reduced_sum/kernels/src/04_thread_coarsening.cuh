#include<iostream>
using namespace std;

template<size_t NUM_THREADS>
__global__ void batched_thread_coarsening(
  float* __restrict__ Y,
  float* const __restrict__ X,
  size_t num_elements_per_batch
) {
  __shared__ float shared_data[NUM_THREADS];
  X += num_elements_per_batch * blockIdx.x;

  size_t const num_elements_per_thread = (num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS;
  float sum = 0.0f;
  for (int i = 0; i < num_elements_per_thread; i++) {
    size_t const offset = threadIdx.x + i * NUM_THREADS;
    if (offset < num_elements_per_batch) {
      sum += X[offset];
    }
  }

  shared_data[threadIdx.x] = sum;

  for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if(threadIdx.x < stride) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
  }
   
  
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = shared_data[0];
  }
}