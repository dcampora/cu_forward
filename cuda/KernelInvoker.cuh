#pragma once

#include <iostream>
#include "Definitions.cuh"
#include "SearchByTriplet.cuh"
#include "Tools.cuh"
#include "../src/Logger.h"
#include "../src/Common.h"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <map>
#include <stdint.h>
#include <assert.h>

void printOutSensorHits(const EventInfo& info, int sensorNumber, int* prevs, int* nexts);
void printOutAllSensorHits(const EventInfo& info, int* prevs, int* nexts);
void printInfo(const EventInfo& info, int numberOfSensors, int numberOfHits);
void printTrack(const EventInfo& info, Track* tracks, const int trackNumber, const std::map<int, int>& zhit_to_module, std::ofstream& outstream);
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module);

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
);

struct EventBeginning {
  int numberOfSensors;
  int numberOfHits;
};
