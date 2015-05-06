#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"
// #include "KernelFits.cuh"

__device__ float fitHits(const Hit& h0, const Hit& h1, const Hit& h2);
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float h1_z, const Hit& h2);

__global__ void fillCandidates(int* const dev_hit_candidate_pointer,
    int* const dev_hit_candidates, const char* const dev_input,
    int* const dev_event_offsets, int* const dev_hit_offsets);

__global__ void searchByTriplet(Track* const dev_tracks, const char* const dev_input,
    int* const dev_tracks_to_follow,
    bool* const dev_hit_used, int* const dev_hit_candidates, int* const dev_atomicsStorage, Track* const dev_tracklets,
    int* const dev_weak_tracks, int* const dev_event_offsets, int* const dev_hit_offsets, float* const dev_best_fits);

#endif
