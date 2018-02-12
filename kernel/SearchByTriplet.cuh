#pragma once

#include "Definitions.cuh"

__device__ float fitHitToTrack(
  const float tx,
  const float ty,
  const Hit& h0,
  const float h1_z,
  const Hit& h2
);

__device__ void processModules(
#if USE_SHARED_FOR_HITS
  float* sh_hit_x,
  float* sh_hit_y,
#endif
  Sensor* sensor_data,
  const int starting_sensor,
  const int stride,
  bool* hit_used,
  const int* hit_candidates,
  const int* hit_h2_candidates,
  const int number_of_sensors,
  const int* sensor_hitStarts,
  const int* sensor_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* sensor_Zs,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* sh_hit_lastPointer,
  unsigned int* max_numhits_to_process,
  unsigned int* tracks_insertPointer,
  const int blockDim_sh_hit,
  const int blockDim_product,
  int* tracks_to_follow,
  int* weak_tracks,
  Track* tracklets,
  float* best_fits,
  Track* tracks,
  const int number_of_hits,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
);

__device__ void fillCandidates(
  int* hit_candidates,
  int* hit_h2_candidates,
  const int number_of_sensors,
  const int* sensor_hitStarts,
  const int* sensor_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* sensor_Zs
);

__device__ void trackForwarding(
#if USE_SHARED_FOR_HITS
  float* sh_hit_x,
  float* sh_hit_y,
#endif
  const float* hit_Xs,
  const float* hit_Ys,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const int blockDim_sh_hit,
  const Sensor* sensor_data,
  const unsigned int diff_ttf,
  const int blockDim_product,
  int* tracks_to_follow,
  int* weak_tracks,
  const unsigned int prev_ttf,
  Track* tracklets,
  Track* tracks,
  const int number_of_hits,
  const int first_sensor,
  const float* sensor_Zs,
  const int* sensor_hitStarts,
  const int* sensor_hitNums
);

__device__ void trackSeeding(
  const float* hit_Xs,
  const float* hit_Ys,
  const Sensor* sensor_data,
  const int* hit_candidates,
  unsigned int* max_numhits_to_process,
  bool* hit_used,
  const int* hit_h2_candidates,
  const int blockDim_sh_hit,
  float* best_fits,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  int* tracks_to_follow,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
);

__global__ void searchByTriplet(
  Track* dev_tracks,
  const char* dev_input,
  int* dev_tracks_to_follow,
  bool* dev_hit_used,
  int* dev_atomicsStorage,
  Track* dev_tracklets,
  int* dev_weak_tracks,
  int* dev_event_offsets,
  int* dev_hit_offsets,
  float* dev_best_fits,
  int* dev_hit_candidates,
  int* dev_hit_h2_candidates,
  unsigned int* dev_rel_indices
);
