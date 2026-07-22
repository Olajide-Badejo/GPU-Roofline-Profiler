// GEMV: y := A * x, with A being m by n in row major order.
//
// One block per output row. The threads of a block stride along that row, so the
// reads of A are coalesced across the warp, and the partial products are folded
// with the same shared memory tree the reduction kernel uses.
//
// Each element of A is read exactly once and feeds exactly one multiply add, so
// the arithmetic intensity is fixed at roughly 2 FLOPs per 4 bytes no matter how
// the kernel is written. That is why GEMV sits mid axis and cannot be optimized
// up to the compute ceiling: the ratio is a property of the operation.

#include "kernels.hpp"

#include "cuda_check.hpp"

namespace roofline {
namespace {

__global__ void gemv_kernel(const float* __restrict__ a,
                            const float* __restrict__ x, float* __restrict__ y,
                            int m, int n) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int tid = threadIdx.x;
    const float* row_ptr = a + static_cast<size_t>(row) * n;

    float sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        sum += row_ptr[j] * x[j];
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
        y[row] = sdata[0];
    }
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

void launch_gemv(const float* a, const float* x, float* y, int m, int n,
                 int block_size, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    // Same constraint and same reasoning as the reduction: the tree halving
    // assumes a power of two width.
    if (!is_power_of_two(block_size)) {
        std::fprintf(stderr, "gemv block_size must be a power of two, got %d\n",
                     block_size);
        std::exit(EXIT_FAILURE);
    }
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);
    gemv_kernel<<<m, block_size, shared_bytes, stream>>>(a, x, y, m, n);
    CUDA_CHECK_KERNEL();
}

}  // namespace roofline
