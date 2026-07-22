cmake_minimum_required(VERSION 3.24)

# The CUDA architecture has to be chosen BEFORE project() enables the CUDA
# language, because project() installs a default of its own and a later
# "if(NOT DEFINED ...)" guard would never fire. Getting this wrong is quiet and
# expensive: the build silently targets the default architecture (sm_75 here)
# instead of the card actually in the machine.
#
# Order of preference: an explicit CUDAARCHS from the environment, then "native"
# to detect the installed GPU, then 120 (Blackwell, the RTX 5070) as the
# fallback for a machine with no GPU, such as a compile only CI runner.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
  elseif(DEFINED ENV{CUDA_NO_GPU} OR DEFINED ENV{CI})
    set(CMAKE_CUDA_ARCHITECTURES 120)
  else()
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

project(gpu_roofline_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# yaml-cpp 0.8.0 declares a cmake_minimum_required below 3.5, and CMake 4
# removed compatibility with those. This tells CMake to treat the old
# declaration as 3.5 so the dependency still configures. It is scoped by being
# set just before this dependency is populated, and it affects only the fetched
# subproject, not this one.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)
unset(CMAKE_POLICY_VERSION_MINIMUM CACHE)

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
