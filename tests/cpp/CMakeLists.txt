# GPU correctness tests. These need a real device, so they are run locally with
# ctest and are explicitly not part of the CPU only CI job.

add_executable(kernel_tests test_kernels.cu test_gemm.cu)
target_include_directories(kernel_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernel_tests PRIVATE roofline_kernels GTest::gtest_main)
set_target_properties(kernel_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(GoogleTest)
gtest_discover_tests(kernel_tests PROPERTIES LABELS gpu)
