#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"

__device__ float fitHits(Hit& h0, Hit& h1, Hit& h2);
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const Hit& h2);

__device__ void acceptTrack(Track& t, TrackFit& fit, const Hit& h0, const Hit& h1, const int h0_num, const int h1_num);
__device__ void updateTrack(Track& t, TrackFit& fit, const Hit& h1, const int h1_num);
__device__ void updateTrackCoords(Track& t, TrackFit& fit);

__global__ void prepareData(char* input, int* _tracks_to_follow_q1, int* _tracks_to_follow_q2,
	bool* used, int* _atomicsStorage, Track* dev_tracklets, int* dev_weak_tracks);
__global__ void searchByTriplet(Track* tracks);

#endif
