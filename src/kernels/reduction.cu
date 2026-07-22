// Shared memory tree reduction: out[0] := sum(in[0..n)).
//
// Two stages. The first launches a fixed, modest number of blocks, each of which
// walks the input with a grid stride loop and reduces its running total through
// shared memory into one partial per block. The second stage reduces those
// partials to a single value with one block. Two launches beat a single launch
// plus a host copy, and they keep the partial count bounded regardless of n.
//
// The grid stride loop is also what makes non power of two sizes work: each
// thread only ever accumulates indices that exist, so the tail needs no special
// case. The tree itself does require a power of two block size, which is checked
// at launch rather than assumed, because a silently wrong sum is exactly the
// kind of bug this kernel is prone to.

#include "kernels.hpp"

#include "cuda_check.hpp"

namespace roofline {
namespace {

// Bounded so the second stage always fits in one block's grid stride loop.
constexpr int kMaxReductionBlocks = 1024;

__global__ void reduce_kernel(const float* __restrict__ in,
                              float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<unsigned int>(tid) < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

int reduction_scratch_elements() { return kMaxReductionBlocks; }

void launch_reduction(const float* in, float* out, float* scratch, int n,
                      int block_size, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    // The tree halving step assumes a power of two block width. Fail loudly
    // rather than return a quietly wrong sum.
    if (!is_power_of_two(block_size)) {
        std::fprintf(stderr,
                     "reduction block_size must be a power of two, got %d\n",
                     block_size);
        std::exit(EXIT_FAILURE);
    }

    int blocks = (n + block_size - 1) / block_size;
    if (blocks > kMaxReductionBlocks) {
        blocks = kMaxReductionBlocks;
    }
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    reduce_kernel<<<blocks, block_size, shared_bytes, stream>>>(in, scratch, n);
    CUDA_CHECK_KERNEL();

    // Second stage: one block folds the per block partials into out[0].
    reduce_kernel<<<1, block_size, shared_bytes, stream>>>(scratch, out, blocks);
    CUDA_CHECK_KERNEL();
}

}  // namespace roofline
