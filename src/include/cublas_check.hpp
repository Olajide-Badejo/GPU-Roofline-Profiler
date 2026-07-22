// Error checking wrapper for cuBLAS, kept in its own header so the many
// translation units that never touch cuBLAS do not pull in its header. Same
// contract as CUDA_CHECK: report the call, the file, the line, and a decoded
// status, then abort.

#ifndef ROOFLINE_CUBLAS_CHECK_HPP
#define ROOFLINE_CUBLAS_CHECK_HPP

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

namespace roofline {

// cuBLAS has no strerror equivalent in older versions, so the common statuses
// are decoded by hand. Anything unrecognized still reports its numeric value.
inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "unknown cublasStatus_t";
    }
}

inline void cublas_check_failed(cublasStatus_t status, const char* expr,
                                const char* file, int line) {
    std::fprintf(stderr, "cuBLAS error %d (%s) at %s:%d\n  in: %s\n",
                 static_cast<int>(status), cublas_status_string(status), file,
                 line, expr);
    std::exit(EXIT_FAILURE);
}

}  // namespace roofline

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t status_ = (expr);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
            ::roofline::cublas_check_failed(status_, #expr, __FILE__,          \
                                            __LINE__);                         \
        }                                                                      \
    } while (0)

#endif  // ROOFLINE_CUBLAS_CHECK_HPP
