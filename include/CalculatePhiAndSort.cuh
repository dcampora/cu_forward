#pragma once

#include "Definitions.cuh"

__device__ void calculatePhi(
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  unsigned short* hit_permutations
);

__device__ void sortByPhi(
  const unsigned int number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  unsigned int* hit_IDs,
  int32_t* hit_temp,
  unsigned short* hit_permutation
);

__device__ void calculatePhiAndSort(
  const char* dev_input,
  unsigned int* dev_event_offsets,
  unsigned int* dev_hit_offsets,
  float* dev_hit_phi,
  int32_t* dev_hit_temp,
  unsigned short* dev_hit_permutation
);
