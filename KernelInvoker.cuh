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
#include <iomanip>
#include <map>
#include <stdint.h>
#include <assert.h>

void getMaxNumberOfHits(char*& input, int& maxHits);
void printOutSensorHits(int sensorNumber, int* prevs, int* nexts, std::ostream& logger);
void printOutAllSensorHits(int* prevs, int* nexts, std::ostream& logger);
void printInfo(std::ostream& logger);
void printTrack(Track* tracks, const int trackID, std::ostream& logger,
  const int trackNumber, const std::map<int, int>& zhit_to_module);
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module);

cudaError_t invokeParallelSearch(
    dim3                         numBlocks,
    dim3                         numThreads,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t>       & solution,
    std::ostream               & logger);

#endif
