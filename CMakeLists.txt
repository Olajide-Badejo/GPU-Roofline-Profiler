cmake_minimum_required(VERSION 3.24)

# CUDA is enabled as a project language, which requires a CUDA compiler at
# configure time. On a machine without the toolkit this configure step is
# expected to fail with a clear message; that is the signal to install the
# toolchain, not a bug in this file.
project(gpu_roofline_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto detect the architecture of the installed GPU, falling back to sm_120
# (Blackwell, the RTX 5070) when detection is unavailable, for example on a CI
# box with no GPU doing a compile only build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  include(CheckLanguage)
  if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
  else()
    set(CMAKE_CUDA_ARCHITECTURES "native")
  endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

# Third party dependencies fetched at configure time so the build is self
# contained: yaml-cpp for the sweep config loader, GoogleTest for the CUDA
# correctness tests.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# Kernel and profiling sources are globbed so adding a new kernel family does not
# mean editing this file. Targets are only created once their sources exist, so
# this configures cleanly during the scaffold phase before any kernel is written.
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")
file(GLOB PROFILING_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/profiling/*.cpp"
     "${CMAKE_SOURCE_DIR}/src/profiling/*.cu")

# Only create the library once at least one kernel source exists; a static
# library with no sources cannot determine its link language and fails to
# configure. This keeps the scaffold phase (no .cu files yet) configuring
# cleanly.
if(KERNEL_SOURCES OR PROFILING_SOURCES)
  add_library(roofline_kernels STATIC ${KERNEL_SOURCES} ${PROFILING_SOURCES})
  target_include_directories(roofline_kernels PUBLIC
    "${CMAKE_SOURCE_DIR}/src/include")
  target_link_libraries(roofline_kernels PUBLIC
    CUDA::cudart CUDA::cublas CUDA::nvml yaml-cpp::yaml-cpp)
  set_target_properties(roofline_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    add_executable(roofline_profiler "${CMAKE_SOURCE_DIR}/src/main.cpp")
    target_link_libraries(roofline_profiler PRIVATE roofline_kernels)
  endif()
else()
  message(STATUS "no kernel sources yet; skipping library and driver targets")
endif()

enable_testing()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
  add_subdirectory(tests/cpp)
endif()
