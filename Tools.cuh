
/**
 * Tools.h
 */

#ifndef TOOLS
#define TOOLS 1

#include "CudaException.h"
 
#include "cuda_runtime.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>
#include <stdint.h>

#define cudaCheck(stmt) {                                    \
    cudaError_t err = stmt;                                  \
    if (err != cudaSuccess){                                 \
        std::cerr << "Failed to run " << #stmt << std::endl; \
        std::cerr << cudaGetErrorString(err) << std::endl;   \
        return err;                                          \
    }                                                        \
}

template <class T>
std::string toString(T t){
    std::stringstream ss;
    std::string s;
    ss << t;
    ss >> s;
    return s;
}

void setHPointersFromInput(uint8_t * input, size_t size);
void mergeSolutions(const std::vector<std::vector<char> >& solutions, std::vector<char>& output);

// A non-efficient implementation that does what I need
void preorder_by_x(std::vector<const std::vector<uint8_t>* > & input);
void quicksort (float* a, float* b, float* c, unsigned int* d, int start, int end);
int divide (float* a, float* b, float* c, unsigned int* d, int first, int last);
template<typename T> void swap (T& a, T& b);

std::map<std::string, float> calcResults(std::vector<float>& times);
float float_max();

#endif
