// SAXPY: y := a * x + y.
//
// The left anchor of the roofline. Two floating point operations per element
// against twelve bytes of traffic (read x, read y, write y) puts its arithmetic
// intensity at 1/6 FLOP per byte, far into the memory bound region. If this
// kernel does not approach the measured copy bandwidth, the harness or the
// bandwidth microbenchmark is wrong, and there is no point looking at any GEMM
// until that is resolved.

#include "kernels.hpp"

#include "cuda_check.hpp"

namespace roofline {
namespace {

// A grid stride loop rather than one element per thread, so a single launch
// configuration covers every problem size in the sweep without the grid
// exceeding what the device will accept on the largest sizes.
__global__ void saxpy_kernel(float a, const float* __restrict__ x,
                             float* __restrict__ y, int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

}  // namespace

void launch_saxpy(float a, const float* x, float* y, int n, int block_size,
                  cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    const int blocks = (n + block_size - 1) / block_size;
    saxpy_kernel<<<blocks, block_size, 0, stream>>>(a, x, y, n);
    CUDA_CHECK_KERNEL();
}

}  // namespace roofline
