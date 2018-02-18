#pragma once

#include <dirent.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <stdint.h>
#include "Logger.h"
#include "Common.h"
#include "Definitions.cuh"

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};

void readFileIntoVector(
  const std::string& foldername,
  std::vector<uint8_t> & output
);

std::vector<std::vector<uint8_t>> readFolder(
  const std::string& foldername,
  int fileNumber
);

void statistics(
  const std::vector<std::vector<uint8_t>>& input
);

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

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
