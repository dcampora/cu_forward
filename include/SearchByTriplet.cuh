#pragma once

#include "Definitions.cuh"
#include "CalculatePhiAndSort.cuh"

__device__ float fitHitToTrack(
  const float tx,
  const float ty,
  const Hit& h0,
  const float h1_z,
  const Hit& h2
);

__device__ void processModules(
  Module* module_data,
  float* shared_best_fits,
  const unsigned int starting_module,
  const unsigned int stride,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const unsigned int number_of_modules,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* module_Zs,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* tracks_to_follow,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  const unsigned int number_of_hits,
  unsigned short* h1_rel_indices,
  unsigned int* local_number_of_hits
);

__device__ void fillCandidates(
  short* h0_candidates,
  short* h2_candidates,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Phis
);

__device__ void trackForwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const Module* module_data,
  const unsigned int diff_ttf,
  unsigned int* tracks_to_follow,
  unsigned int* weak_tracks,
  const unsigned int prev_ttf,
  Track* tracklets,
  Track* tracks,
  const unsigned int number_of_hits,
  const unsigned int first_module,
  const float* module_Zs,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums
);

__device__ void trackSeeding(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  bool* hit_used,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  unsigned int* tracks_to_follow,
  unsigned short* h1_rel_indices,
  unsigned int* local_number_of_hits
);

__device__ void trackSeedingFirst(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  unsigned int* tracks_to_follow
);

__device__ void weakTracksAdder(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  bool* hit_used
);

__device__ void weakTracksAdderShared(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  bool* hit_used
);

__global__ void searchByTriplet(
  Track* dev_tracks,
  const char* dev_input,
  unsigned int* dev_tracks_to_follow,
  bool* dev_hit_used,
  int* dev_atomicsStorage,
  Track* dev_tracklets,
  unsigned int* dev_weak_tracks,
  unsigned int* dev_event_offsets,
  unsigned int* dev_hit_offsets,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices,
  float* dev_hit_phi,
  int32_t* dev_hit_temp,
  unsigned short* dev_hit_permutation
);
