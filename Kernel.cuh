#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"

__device__ float fitHits(Hit& h0, Hit& h1, Hit& h2);
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const Hit& h2);

__device__ void acceptTrack(Track& t, TrackFit& fit, const Hit& h0, const Hit& h1, const int h0_num, const int h1_num);
__device__ void updateTrack(Track& t, TrackFit& fit, const Hit& h1, const int h1_num);
__device__ void updateTrackCoords(Track& t, TrackFit& fit);

__global__ void prepareData(char* input, int* _prevs, int* _nexts, bool* track_holders);
__global__ void searchByTriplet(Track* tracks, bool* track_holders);

#endif
