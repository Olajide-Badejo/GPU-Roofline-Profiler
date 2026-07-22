// Error checking wrapper for the CUDA runtime. Every runtime call in this
// project goes through it so a failure reports its file, line, and a decoded
// error string and then aborts, instead of corrupting a later result silently
// (spec Section 4, rule 5). The cuBLAS reference kernel uses a sibling
// CUBLAS_CHECK defined next to it, so this header stays free of the cuBLAS
// include for the many translation units that never touch cuBLAS.
//
// Usage:
//     CUDA_CHECK(cudaMalloc(&p, bytes));
//     kernel<<<grid, block>>>(...);
//     CUDA_CHECK_KERNEL();   // checks launch error and, in debug, synchronizes

#ifndef ROOFLINE_CUDA_CHECK_HPP
#define ROOFLINE_CUDA_CHECK_HPP

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace roofline {

// Reports a failed cudaError_t and aborts. Kept out of line of the macro body so
// the macro expands to a single cheap comparison on the success path.
inline void cuda_check_failed(cudaError_t status, const char* expr,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n  in: %s\n",
                 static_cast<int>(status), cudaGetErrorString(status), file,
                 line, expr);
    std::exit(EXIT_FAILURE);
}

}  // namespace roofline

// Wrap any runtime call returning cudaError_t.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t status_ = (expr);                                          \
        if (status_ != cudaSuccess) {                                         \
            ::roofline::cuda_check_failed(status_, #expr, __FILE__, __LINE__); \
        }                                                                      \
    } while (0)

// Check a kernel launch. Always catches launch-time errors (bad launch config);
// in a debug build it also synchronizes to surface asynchronous faults at the
// launch site rather than at some unrelated later call.
#if defined(NDEBUG)
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
    } while (0)
#else
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)
#endif

#endif  // ROOFLINE_CUDA_CHECK_HPP
