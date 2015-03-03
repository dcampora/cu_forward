#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"

__device__ float fitHits(Hit& h0, Hit& h1, Hit& h2);
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float& h1_z, const Hit& h2);

// __device__ void acceptTrack(Track& t, TrackFit& fit, const Hit& h0, const Hit& h1, const int h0_num, const int h1_num);
// __device__ void updateTrack(Track& t, TrackFit& fit, const Hit& h1, const int h1_num);
// __device__ void updateTrackCoords(Track& t, TrackFit& fit);

__global__ void searchByTriplet(Track* dev_tracks, char* dev_input, int* dev_tracks_to_follow_q1, int* dev_tracks_to_follow_q2,
  bool* dev_hit_used, int* dev_atomicsStorage, Track* dev_tracklets, int* dev_weak_tracks, int* dev_event_offsets, int* dev_hit_offsets);

#endif
