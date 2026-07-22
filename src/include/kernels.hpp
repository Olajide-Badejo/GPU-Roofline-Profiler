// Host side launch wrappers for every kernel in the suite.
//
// The kernels themselves stay in their .cu files; this header exposes only the
// launch functions the driver and the tests call, so nothing outside a
// translation unit needs to know a kernel's block or tile geometry. Tile sizes
// are template parameters inside the .cu files and are selected here by an
// explicit argument, so no magic number is ever copied between files.
//
// Every wrapper launches on the caller's stream and checks the launch through
// CUDA_CHECK_KERNEL. None of them synchronize: the harness owns synchronization
// so it can batch many launches between one pair of CUDA events.

#ifndef ROOFLINE_KERNELS_HPP
#define ROOFLINE_KERNELS_HPP

#include <cuda_runtime.h>

namespace roofline {

// y := a * x + y, over n elements. 2n FLOPs, 3n floats of theoretical traffic
// (read x, read y, write y).
void launch_saxpy(float a, const float* x, float* y, int n, int block_size,
                  cudaStream_t stream = nullptr);

// Sum of in[0..n). Writes the single total to out. Needs a caller supplied
// scratch buffer of at least reduction_scratch_elements() floats. Handles non
// power of two n, which is where tree reductions usually break.
void launch_reduction(const float* in, float* out, float* scratch, int n,
                      int block_size, cudaStream_t stream = nullptr);

// Number of float elements the reduction scratch buffer must hold.
int reduction_scratch_elements();

// out := transpose(in), where in is rows by cols in row major order and out is
// cols by rows. Zero FLOPs, so these isolate the memory system with no
// arithmetic to hide behind.
void launch_transpose_naive(const float* in, float* out, int rows, int cols,
                            int tile_dim, cudaStream_t stream = nullptr);

// Same result, but staged through a padded shared memory tile so that both the
// global read and the global write are coalesced and the transposed shared
// access is free of bank conflicts.
void launch_transpose_tiled(const float* in, float* out, int rows, int cols,
                            int tile_dim, cudaStream_t stream = nullptr);

// y := A * x, with A being m by n in row major order. 2*m*n FLOPs. Each matrix
// element is read once and feeds one multiply add, so reuse is low and this
// sits mid axis, between the streaming kernels and the GEMM ladder.
void launch_gemv(const float* a, const float* x, float* y, int m, int n,
                 int block_size, cudaStream_t stream = nullptr);

// The GEMM ladder. All compute C := A * B with A being m by k, B being k by n,
// and C being m by n, every matrix row major. 2*m*n*k FLOPs.
//
// Each rung raises reuse over the one before it, and the roofline shows
// arithmetic intensity and achieved performance climbing together as it does.

// Every operand read straight from global memory, so each value is refetched
// many times. The memory bound bottom of the ladder.
void launch_gemm_naive(const float* a, const float* b, float* c, int m, int n,
                       int k, int tile_dim, cudaStream_t stream = nullptr);

// Tiles of A and B staged in shared memory once and reused across the tile.
void launch_gemm_tiled(const float* a, const float* b, float* c, int m, int n,
                       int k, int tile_dim, cudaStream_t stream = nullptr);

// Each thread computes a small output tile held in registers, raising reuse per
// shared memory load at the cost of register pressure.
void launch_gemm_register_blocked(const float* a, const float* b, float* c,
                                  int m, int n, int k,
                                  cudaStream_t stream = nullptr);

// The register blocked kernel with float4 loads and stores, so each memory
// instruction moves four values. Requires k and n to be multiples of four;
// returns false without launching when they are not, so the caller can skip the
// configuration rather than read past the end of an array.
bool launch_gemm_vectorized(const float* a, const float* b, float* c, int m,
                            int n, int k, cudaStream_t stream = nullptr);

// The vendor reference. Plotted as an honest ceiling, labeled as a library
// rather than as one of my kernels. The handle is created once and reused,
// because creating one per launch would time cuBLAS setup instead of its GEMM.
void gemm_cublas_init();
void gemm_cublas_destroy();
void launch_gemm_cublas(const float* a, const float* b, float* c, int m, int n,
                        int k, cudaStream_t stream = nullptr);

}  // namespace roofline

#endif  // ROOFLINE_KERNELS_HPP
