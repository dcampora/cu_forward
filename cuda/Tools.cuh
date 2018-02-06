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
float float_max();
