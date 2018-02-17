#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <cmath>
#include <stdint.h>
#include "CudaException.h"
#include "KernelInvoker.cuh"
#include "../src/Common.h"

std::map<std::string, float> calcResults(std::vector<float>& times);

void printOutSensorHits(
  const EventInfo& info,
  int sensorNumber,
  int* prevs,
  int* nexts
);

void printOutAllSensorHits(
  const EventInfo& info,
  int* prevs,
  int* nexts
);

void printInfo(
  const EventInfo& info,
  int numberOfSensors,
  int numberOfHits
);

void printTrack(
  const EventInfo& info,
  Track* tracks,
  const int trackNumber,
  std::ofstream& outstream
);

void writeBinaryTrack(
  const unsigned int* hit_IDs,
  const Track& track,
  std::ofstream& outstream
);

cudaError_t checkSorting(
  const std::vector<std::vector<uint8_t>>& input,
  unsigned int acc_hits,
  unsigned short* dev_hit_phi,
  const std::vector<unsigned int>& hit_offsets
);
