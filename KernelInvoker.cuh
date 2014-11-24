#ifndef KERNEL_INVOKER
#define KERNEL_INVOKER 1

#include <iostream>
#include "Definitions.cuh"
#include "Kernel.cuh"
#include "Tools.cuh"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdint.h>

void getMaxNumberOfHits(char*& input, int& maxHits);
void printTrack(Track* tracks, int i, std::ostream& logger);
void printOutSensorHits(int sensorNumber, int* prevs, int* nexts, std::ostream& logger);
void printOutAllSensorHits(int* prevs, int* nexts, std::ostream& logger);
void printInfo(std::ostream& logger);

cudaError_t invokeParallelSearch(
    dim3                         numBlocks,
    dim3                         numThreads,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t>       & solution,
    std::ostream               & logger);

#endif
