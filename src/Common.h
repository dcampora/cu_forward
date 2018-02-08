#pragma once

#include <vector>
#include "cuda_runtime.h"

#define NUMBER_OF_SENSORS 52

#define cudaCheck(stmt) {                                    \
    cudaError_t err = stmt;                                  \
    if (err != cudaSuccess){                                 \
        std::cerr << "Failed to run " << #stmt << std::endl; \
        std::cerr << cudaGetErrorString(err) << std::endl;   \
        return err;                                          \
    }                                                        \
}

// Little struct to facilitate accessing events
struct EventInfo {
  int32_t numberOfSensors;
  int32_t numberOfHits;
  int32_t* sensor_Zs;
  int32_t* sensor_hitStarts;
  int32_t* sensor_hitNums;
  uint32_t* hit_IDs;
  float* hit_Xs;
  float* hit_Ys;

  // Ignore hit_Zs
  // float* hit_Zs;
  
  int size;

  EventInfo() = default;
  EventInfo(const std::vector<uint8_t>& event) {
    uint8_t* input = (uint8_t*) event.data();
    
    numberOfSensors  = *((int32_t*)input); input += sizeof(int32_t);
    numberOfHits     = *((int32_t*)input); input += sizeof(int32_t);
    sensor_Zs        = (int32_t*)input; input += sizeof(int32_t) * numberOfSensors;
    sensor_hitStarts = (int32_t*)input; input += sizeof(int32_t) * numberOfSensors;
    sensor_hitNums   = (int32_t*)input; input += sizeof(int32_t) * numberOfSensors;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - (uint8_t*) event.data();
    // hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;
  }
};
