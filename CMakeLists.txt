# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(nvcomp-zstd-compress LANGUAGES CUDA CXX)

# GPU architectures
set(GPU_ARCHS "75;80;86;89;90")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13")
  set(GPU_ARCHS ${GPU_ARCHS} "70")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
  set(GPU_ARCHS ${GPU_ARCHS} "100;120")
endif()

# C++ standard
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13")
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CUDA_STANDARD 11)
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Find nvCOMP
find_package(nvcomp REQUIRED)

# zstd_nvcomp
add_executable(zstd_nvcomp zstd_nvcomp.cu)
set_property(TARGET zstd_nvcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_link_libraries(zstd_nvcomp PRIVATE nvcomp::nvcomp CUDA::cudart_static)