cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cuForward CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

find_package(CUDA REQUIRED)  

set(CUDA_HOST_COMPILER "g++")

set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS "-std=c++11;-O3;-g;--expt-relaxed-constexpr; --generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=128;")

cuda_add_executable(cuForward src/main.cpp src/Tools.cpp src/Logger.cpp cuda/CudaException.cpp cuda/Tools.cu cuda/Kernel.cu cuda/KernelInvoker.cu)
