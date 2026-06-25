#include<iostream>
using namespace std;


template<size_t NUM_THREADS>
__global__ void batched_warp_shuffle(
  float* __restrict__ Y,
  float* const __restrict__ X,
  size_t num_elements_per_batch
) {
  __shared__ float shared_data[NUM_THREADS];
  X += num_elements_per_batch * blockIdx.x;

  size_t const num_elements_per_thread = (num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS;
  size_t const NUM_WARPS = NUM_THREADS / 32;
  float sum = 0.0f;
  for (int i = 0; i < num_elements_per_thread; i++) {
    size_t const offset = threadIdx.x + i * NUM_THREADS;
    if (offset < num_elements_per_batch) {
      sum += X[offset];
    }
  }

  constexpr unsigned int FULL_MASK = 0xffffffff;
  for (int offset = 16; offset > 0; offset/=2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }

  if (threadIdx.x % 32 == 0) {
    shared_data[threadIdx.x/32] = sum;
  }
  __syncthreads();


  unsigned int const active_thread_mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_WARPS);
  if (threadIdx.x < NUM_WARPS) {
    sum = shared_data[threadIdx.x];
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(active_thread_mask, sum, offset);
    }
  }
  
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = shared_data[0];
  }
}
