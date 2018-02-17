#pragma once

#include "Definitions.cuh"
#include "SearchByTriplet.cuh"
#include "CalculatePhiAndSort.cuh"
#include "Tools.cuh"
#include "Logger.h"
#include "Common.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <map>
#include <stdint.h>
#include <assert.h>

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
);
