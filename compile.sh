#/bin/bash

# nvcc -ccbin /usr/bin/clang -arch=sm_30 main.cpp kernel.cu kernelInvoker.cu Tools.cpp
nvcc -arch=sm_30 Kernel.cu KernelInvoker.cu Tools.cu GpuPixelSearchByTriplet.cpp IndependentEntrypoint.cpp Logger.cpp -g -o gpupixel
