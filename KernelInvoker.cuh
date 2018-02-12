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
void printInfo(int numberOfSensors, int numberOfHits);
void printTrack(Track* tracks, const int trackNumber, const std::map<int, int>& zhit_to_module, std::ofstream& outstream);
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module);

cudaError_t invokeParallelSearch(
    const int startingEvent,
    const int eventsToProcess,
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output);

struct EventBeginning {
  int numberOfSensors;
  int numberOfHits;
};

/**
 * @brief Struct to typecast events.
 */
struct EventInfo {
  int32_t numberOfSensors;
  int32_t numberOfHits;
  float* sensor_Zs;
  int32_t* sensor_hitStarts;
  int32_t* sensor_hitNums;
  uint32_t* hit_IDs;
  float* hit_Xs;
  float* hit_Ys;
  float* hit_Zs;
  
  int size;

  EventInfo() = default;
  EventInfo(const std::vector<uint8_t>* const event) {
    uint8_t* input = (uint8_t*) event->data();
    
    numberOfSensors  = *((int32_t*)input); input += sizeof(int32_t);
    numberOfHits     = *((int32_t*)input); input += sizeof(int32_t);
    sensor_Zs        = (float*)input; input += sizeof(float) * numberOfSensors;
    sensor_hitStarts = (int32_t*)input; input += sizeof(int32_t) * numberOfSensors;
    sensor_hitNums   = (int32_t*)input; input += sizeof(int32_t) * numberOfSensors;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - (uint8_t*) event->data();
  }
};

#endif
