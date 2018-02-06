cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(clForward CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

find_package(CUDA REQUIRED)  

set(CUDA_HOST_COMPILER "g++")

set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS "-std=c++11;-O3;-g;--expt-relaxed-constexpr; --generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=28;--gpu-architecture=compute_60;--gpu-code=sm_60;")

cuda_add_executable(clForward src/CudaException.cpp src/GpuPixelSearchByTriplet.cpp src/IndependentEntrypoint.cpp src/Logger.cpp src/Tools.cu src/Kernel.cu src/KernelInvoker.cu)
