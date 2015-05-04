#ifndef CUDA_KERNELFITS
#define CUDA_KERNELFITS 1

#include "Definitions.cuh"

__device__ float fitHits(const Hit& h0, const Hit& h1, const Hit& h2);
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float h1_z, const Hit& h2);
__device__ void fillCandidates(int* const hit_candidates, const int no_sensors, 
  const int* const sensor_hitStarts, const int* const sensor_hitNums,
  const float* const hit_Xs, const float* const hit_Ys, const float* const hit_Zs);

#endif
