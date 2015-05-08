#!/bin/bash

for i in `seq 16 64`; do
  sed -i 's/^NVCC :=.*$/NVCC := $\(CUDA_PATH\)\/bin\/nvcc -O2 --maxrregcount='${i}' -std=c++11 -ccbin $\(GCC\)/' Makefile
  make clean
  make
  ./run_2048.sh > results/2048/$i.out
done
