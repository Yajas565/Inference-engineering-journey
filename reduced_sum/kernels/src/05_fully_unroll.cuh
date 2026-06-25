#include<iostream>
using namespace std;

template<size_t NUM_THREADS>
__device__ void warp_reduce(volatile float* shared_data, size_t thread_idx) {
  if (NUM_THREADS >= 64) shared_data[thread_idx] += shared_data[thread_idx + 32];
  if (NUM_THREADS >= 32) shared_data[thread_idx] += shared_data[thread_idx + 16];
  if (NUM_THREADS >= 16) shared_data[thread_idx] += shared_data[thread_idx + 8];
  if (NUM_THREADS >= 8) shared_data[thread_idx] += shared_data[thread_idx + 4];
  if (NUM_THREADS >= 4) shared_data[thread_idx] += shared_data[thread_idx + 2];
  if (NUM_THREADS >= 2) shared_data[thread_idx] += shared_data[thread_idx + 1];
}


template<size_t NUM_THREADS, size_t NUM_THREADS_PER_WARP>
__global__ void batched_fully_unroll(
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
  __syncthreads();

  if (NUM_THREADS == 1024) {
    if (threadIdx.x < 512) shared_data[threadIdx.x] += shared_data[threadIdx.x + 512];
    __syncthreads();
  }
  if (NUM_THREADS >= 512) {
    if (threadIdx.x < 256) shared_data[threadIdx.x] += shared_data[threadIdx.x + 256];
    __syncthreads();
  }
  if (NUM_THREADS >= 256) {
    if (threadIdx.x < 128) shared_data[threadIdx.x] += shared_data[threadIdx.x + 128];
    __syncthreads();
  }
  if (NUM_THREADS >= 128) {
    if (threadIdx.x < 64) shared_data[threadIdx.x] += shared_data[threadIdx.x + 64];
    __syncthreads();
  }

  if (threadIdx.x < NUM_THREADS_PER_WARP) 
    warp_reduce<NUM_THREADS>(shared_data, threadIdx.x);
   
  
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = shared_data[0];
  }
}