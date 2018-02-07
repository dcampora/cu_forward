cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cuForward CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -std=c++11")

find_package(CUDA REQUIRED)  

set(CUDA_HOST_COMPILER "g++")

set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS "-std=c++11;-O2;-g;--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;--gpu-architecture=compute_60;--gpu-code=sm_60;")

cuda_add_executable(cuForward src/main.cpp src/Tools.cpp src/Logger.cpp cuda/CudaException.cpp cuda/Tools.cu cuda/TrackForwarding.cu cuda/TrackSeeding.cu cuda/FillCandidates.cu cuda/KernelInvoker.cu cuda/SearchByTriplet.cu)
