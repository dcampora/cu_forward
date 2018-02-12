#pragma once

#include "Definitions.cuh"

__device__ float fitHitToTrack(
  const float tx,
  const float ty,
  const Hit& h0,
  const float h0_z,
  const float h1_z,
  const Hit& h2,
  const float h2_z
);

__device__ void fillCandidates(
  int* h0_candidates,
  int* h2_candidates,
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
#if USE_SHARED_FOR_HITS
  float* sh_hit_x,
  float* sh_hit_y,
#endif
  const float* hit_Xs,
  const float* hit_Ys,
  const Sensor* sensor_data,
  int* h0_candidates,
  unsigned int* max_numhits_to_process,
  bool* hit_used,
  int* h2_candidates,
  const int blockDim_sh_hit,
  float* best_fits,
  unsigned int* best_h0s,
  unsigned int* best_h2s,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  int* tracks_to_follow,
  unsigned int* local_number_of_hits,
  unsigned int* local_unused_hits,
  unsigned int number_of_hits
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
  int* h0_candidates,
  int* h2_candidates,
  const int number_of_sensors,
  const int* sensor_hitStarts,
  const int* sensor_hitNums,
  const float* sensor_Zs,
  const float* hit_Xs,
  const float* hit_Ys,
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
  unsigned int* best_h0s,
  unsigned int* best_h2s,
  Track* tracks,
  const int number_of_hits,
  unsigned int* local_number_of_hits,
  unsigned int* local_unused_hits
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
  unsigned int* dev_best_h0s,
  unsigned int* dev_best_h2s,
  int* dev_h0_candidates,
  int* dev_h2_candidates,
  unsigned int* dev_local_unused_hits
);
