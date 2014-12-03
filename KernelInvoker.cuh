#ifndef KERNEL_INVOKER
#define KERNEL_INVOKER 1

#include <iostream>
#include "Definitions.cuh"
#include "Kernel.cuh"
#include "Tools.cuh"
#include "Logger.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <map>
#include <stdint.h>
#include <assert.h>

void getMaxNumberOfHits(char*& input, int& maxHits);
void printOutSensorHits(int sensorNumber, int* prevs, int* nexts);
void printOutAllSensorHits(int* prevs, int* nexts);
void printInfo();
void printTrack(Track* tracks, const int trackID,
  const int trackNumber, const std::map<int, int>& zhit_to_module);
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module);

cudaError_t invokeParallelSearch(
    dim3                         numBlocks,
    dim3                         numThreads,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t>       & solution);

#endif
