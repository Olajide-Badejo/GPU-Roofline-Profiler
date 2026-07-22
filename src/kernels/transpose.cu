// Matrix transpose, naive and tiled.
//
// These do no floating point work at all, which is the point: with no arithmetic
// to hide behind, the only thing separating the two versions is how they touch
// memory. The naive kernel reads coalesced and writes with a large stride. The
// tiled kernel stages a tile in shared memory so both the global read and the
// global write are coalesced.
//
// The tile is declared one column wider than it is tall, tile[TILE][TILE + 1].
// Shared memory is banked by 32 consecutive 4 byte words, so an unpadded
// TILE by TILE tile makes every thread in a column hit the same bank on the
// transposed read, serializing the warp. One pad word per row shifts each row by
// a bank and the conflicts disappear. That single "+ 1" is the whole trick, and
// the naive versus tiled delta is the cleanest demonstration in the suite.

#include "kernels.hpp"

#include "cuda_check.hpp"

namespace roofline {
namespace {

// in is rows by cols (row major); out is cols by rows.
__global__ void transpose_naive_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int rows,
                                       int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        // Read is coalesced across threadIdx.x; the write strides by rows.
        out[static_cast<size_t>(col) * rows + row] =
            in[static_cast<size_t>(row) * cols + col];
    }
}

template <int TILE>
__global__ void transpose_tiled_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int rows,
                                       int cols) {
    __shared__ float tile[TILE][TILE + 1];

    // Load a tile, coalesced along the input's rows.
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (col < cols && row < rows) {
        tile[threadIdx.y][threadIdx.x] = in[static_cast<size_t>(row) * cols + col];
    }
    __syncthreads();

    // Write it back transposed, coalesced along the output's rows. The block's
    // x and y origins swap, and the shared read is transposed instead of the
    // global write.
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if (col < rows && row < cols) {
        out[static_cast<size_t>(row) * rows + col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

}  // namespace

void launch_transpose_naive(const float* in, float* out, int rows, int cols,
                            int tile_dim, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const dim3 block(tile_dim, tile_dim);
    const dim3 grid((cols + tile_dim - 1) / tile_dim,
                    (rows + tile_dim - 1) / tile_dim);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    CUDA_CHECK_KERNEL();
}

void launch_transpose_tiled(const float* in, float* out, int rows, int cols,
                            int tile_dim, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const dim3 grid((cols + tile_dim - 1) / tile_dim,
                    (rows + tile_dim - 1) / tile_dim);

    // The tile size is a template parameter so the shared array is sized at
    // compile time; the sweep selects among the instantiations below.
    switch (tile_dim) {
        case 16: {
            const dim3 block(16, 16);
            transpose_tiled_kernel<16><<<grid, block, 0, stream>>>(in, out, rows,
                                                                   cols);
            break;
        }
        case 32: {
            const dim3 block(32, 32);
            transpose_tiled_kernel<32><<<grid, block, 0, stream>>>(in, out, rows,
                                                                   cols);
            break;
        }
        default:
            std::fprintf(stderr, "unsupported transpose tile_dim %d\n", tile_dim);
            std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK_KERNEL();
}

}  // namespace roofline
