#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"
#include "KernelFits.cuh"

__global__ void searchByTriplet(Track* const dev_tracks, const char* const dev_input,
    int* const dev_tracks_to_follow_q1,
    bool* const dev_hit_used, int* const dev_hit_candidates, int* const dev_atomicsStorage, Track* const dev_tracklets,
    int* const dev_weak_tracks, int* const dev_event_offsets, int* const dev_hit_offsets, float* const dev_best_fits);

#endif
