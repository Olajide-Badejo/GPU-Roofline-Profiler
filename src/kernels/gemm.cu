// The GEMM ladder: C := A * B, with A m by k, B k by n, C m by n, all row major.
//
// Four hand written rungs plus the vendor reference. Each rung raises the reuse
// of every value fetched from memory, which is what moves it right along the
// arithmetic intensity axis and up toward the compute ceiling.
//
//   naive             every operand reread from global memory per output
//   tiled             tiles of A and B staged once in shared memory
//   register blocked  each thread owns a TM by TN output tile held in registers
//   vectorized        the same, with float4 global loads and stores
//   cuBLAS            the vendor library, for reference only
//
// The register blocked geometry is the one non obvious part. A block computes a
// BM by BN patch of C using a BK deep slice of A and B at a time. Each thread
// owns a TM by TN sub tile, so a block needs (BM/TM) by (BN/TN) threads. With
// the constants below that is 16 by 16, or 256 threads, each holding 16 running
// sums in registers. The payoff is that one shared memory load of an A value
// feeds TN multiply adds and one B value feeds TM of them, so the shared traffic
// per useful FLOP drops by roughly a factor of TM, at the cost of registers.

#include "kernels.hpp"

#include "cublas_check.hpp"
#include "cuda_check.hpp"

namespace roofline {
namespace {

// Register blocking geometry. Chosen so a block is 256 threads and both shared
// tiles stay well inside the 48 KB per block limit:
// A tile 64*16*4 = 4 KB, B tile 16*64*4 = 4 KB.
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kThreadsPerBlock = (kBM / kTM) * (kBN / kTN);

__global__ void gemm_naive_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += a[static_cast<size_t>(row) * k + i] *
               b[static_cast<size_t>(i) * n + col];
    }
    c[static_cast<size_t>(row) * n + col] = acc;
}

template <int TILE>
__global__ void gemm_tiled_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int m, int n, int k) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        // Zero fill out of range lanes so the inner product below needs no
        // boundary test and every thread can take part in the syncthreads.
        as[ty][tx] = (row < m && a_col < k)
                         ? a[static_cast<size_t>(row) * k + a_col]
                         : 0.0f;
        bs[ty][tx] = (b_row < k && col < n)
                         ? b[static_cast<size_t>(b_row) * n + col]
                         : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            acc += as[ty][i] * bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[static_cast<size_t>(row) * n + col] = acc;
    }
}

__global__ __launch_bounds__(kThreadsPerBlock) void gemm_register_blocked_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k) {
    __shared__ float as[kBM][kBK];
    __shared__ float bs[kBK][kBN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;

    float acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int tiles = (k + kBK - 1) / kBK;
    for (int t = 0; t < tiles; ++t) {
        // Cooperative load of both tiles. The linear thread index walks the tile
        // so the loads stay coalesced regardless of the thread block shape.
        for (int idx = tid; idx < kBM * kBK; idx += kThreadsPerBlock) {
            const int r = idx / kBK;
            const int cc = idx % kBK;
            const int gr = block_row + r;
            const int gc = t * kBK + cc;
            as[r][cc] = (gr < m && gc < k)
                            ? a[static_cast<size_t>(gr) * k + gc]
                            : 0.0f;
        }
        for (int idx = tid; idx < kBK * kBN; idx += kThreadsPerBlock) {
            const int r = idx / kBN;
            const int cc = idx % kBN;
            const int gr = t * kBK + r;
            const int gc = block_col + cc;
            bs[r][cc] = (gr < k && gc < n)
                            ? b[static_cast<size_t>(gr) * n + gc]
                            : 0.0f;
        }
        __syncthreads();

        // The reuse that pays for the whole scheme: TM + TN shared loads feed
        // TM * TN multiply adds.
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a_reg[kTM];
            float b_reg[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                a_reg[i] = as[threadIdx.y * kTM + i][kk];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                b_reg[j] = bs[kk][threadIdx.x * kTN + j];
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int gr = block_row + threadIdx.y * kTM + i;
        if (gr >= m) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int gc = block_col + threadIdx.x * kTN + j;
            if (gc < n) {
                c[static_cast<size_t>(gr) * n + gc] = acc[i][j];
            }
        }
    }
}

// Same blocking, but every global access moves a float4. Requires k and n to be
// multiples of 4 so that a four wide load starting at a multiple of 4 never runs
// past the end of a row; the launcher enforces that and refuses otherwise.
__global__ __launch_bounds__(kThreadsPerBlock) void gemm_vectorized_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k) {
    __shared__ float as[kBM][kBK];
    __shared__ float bs[kBK][kBN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;

    float acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int tiles = (k + kBK - 1) / kBK;
    for (int t = 0; t < tiles; ++t) {
        // One float4 per iteration instead of one float, so the tile load issues
        // a quarter as many memory instructions.
        for (int idx = tid; idx < (kBM * kBK) / 4; idx += kThreadsPerBlock) {
            const int elem = idx * 4;
            const int r = elem / kBK;
            const int cc = elem % kBK;
            const int gr = block_row + r;
            const int gc = t * kBK + cc;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gr < m && gc < k) {
                v = *reinterpret_cast<const float4*>(
                    &a[static_cast<size_t>(gr) * k + gc]);
            }
            as[r][cc + 0] = v.x;
            as[r][cc + 1] = v.y;
            as[r][cc + 2] = v.z;
            as[r][cc + 3] = v.w;
        }
        for (int idx = tid; idx < (kBK * kBN) / 4; idx += kThreadsPerBlock) {
            const int elem = idx * 4;
            const int r = elem / kBN;
            const int cc = elem % kBN;
            const int gr = t * kBK + r;
            const int gc = block_col + cc;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gr < k && gc < n) {
                v = *reinterpret_cast<const float4*>(
                    &b[static_cast<size_t>(gr) * n + gc]);
            }
            bs[r][cc + 0] = v.x;
            bs[r][cc + 1] = v.y;
            bs[r][cc + 2] = v.z;
            bs[r][cc + 3] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a_reg[kTM];
            float b_reg[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                a_reg[i] = as[threadIdx.y * kTM + i][kk];
            }
            *reinterpret_cast<float4*>(b_reg) =
                *reinterpret_cast<const float4*>(&bs[kk][threadIdx.x * kTN]);
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // TN is 4, so each row of the thread's tile is exactly one float4 store.
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int gr = block_row + threadIdx.y * kTM + i;
        const int gc = block_col + threadIdx.x * kTN;
        if (gr < m && gc < n) {
            *reinterpret_cast<float4*>(&c[static_cast<size_t>(gr) * n + gc]) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
    }
}

cublasHandle_t g_cublas_handle = nullptr;

}  // namespace

void launch_gemm_naive(const float* a, const float* b, float* c, int m, int n,
                       int k, int tile_dim, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 block(tile_dim, tile_dim);
    const dim3 grid((n + tile_dim - 1) / tile_dim, (m + tile_dim - 1) / tile_dim);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_CHECK_KERNEL();
}

void launch_gemm_tiled(const float* a, const float* b, float* c, int m, int n,
                       int k, int tile_dim, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 grid((n + tile_dim - 1) / tile_dim, (m + tile_dim - 1) / tile_dim);
    switch (tile_dim) {
        case 16: {
            const dim3 block(16, 16);
            gemm_tiled_kernel<16><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            break;
        }
        case 32: {
            const dim3 block(32, 32);
            gemm_tiled_kernel<32><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            break;
        }
        default:
            std::fprintf(stderr, "unsupported gemm tile_dim %d\n", tile_dim);
            std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK_KERNEL();
}

void launch_gemm_register_blocked(const float* a, const float* b, float* c,
                                  int m, int n, int k, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 block(kBN / kTN, kBM / kTM);
    const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
    gemm_register_blocked_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_CHECK_KERNEL();
}

bool launch_gemm_vectorized(const float* a, const float* b, float* c, int m,
                            int n, int k, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }
    // A four wide load is only safe when the row length is a multiple of four.
    // Report rather than silently reading past a row.
    if ((k % 4) != 0 || (n % 4) != 0) {
        return false;
    }
    const dim3 block(kBN / kTN, kBM / kTM);
    const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
    gemm_vectorized_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_CHECK_KERNEL();
    return true;
}

void gemm_cublas_init() {
    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    }
}

void gemm_cublas_destroy() {
    if (g_cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
        g_cublas_handle = nullptr;
    }
}

void launch_gemm_cublas(const float* a, const float* b, float* c, int m, int n,
                        int k, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    gemm_cublas_init();
    CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS is column major. A row major m by n matrix is the same bytes as a
    // column major n by m one, so computing B^T * A^T in column major yields
    // exactly the row major C = A * B without transposing anything in memory.
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, b, n, a, k, &beta, c, n));
}

}  // namespace roofline
