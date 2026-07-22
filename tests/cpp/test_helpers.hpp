// Shared helpers for the GPU correctness tests: an RAII device buffer and
// deterministic random input. Kept in a header so both test translation units
// use the same definitions rather than drifting copies.

#ifndef ROOFLINE_TEST_HELPERS_HPP
#define ROOFLINE_TEST_HELPERS_HPP

#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace roofline_test {

// Small RAII wrapper so a failing assertion cannot leak device memory.
template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, host.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

   private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Deterministic input: a fixed seed keeps a failure reproducible.
inline std::vector<float> random_vector(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& value : v) {
        value = dist(gen);
    }
    return v;
}

}  // namespace roofline_test

#endif  // ROOFLINE_TEST_HELPERS_HPP
