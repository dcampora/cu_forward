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

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
);
