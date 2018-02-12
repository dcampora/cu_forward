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
  int* sh_hit_process,
  Sensor* sensor_data,
  const int starting_sensor,
  const int stride,
  bool* hit_used,
  const int* const hit_candidates,
  const int* const hit_h2_candidates,
  const int number_of_sensors,
  const int* const sensor_hitStarts,
  const int* const sensor_hitNums,
  const float* const hit_Xs,
  const float* const hit_Ys,
  const float* sensor_Zs,
  unsigned int* const weaktracks_insertPointer,
  unsigned int* const tracklets_insertPointer,
  unsigned int* const ttf_insertPointer,
  unsigned int* const sh_hit_lastPointer,
  unsigned int* const max_numhits_to_process,
  unsigned int* const tracks_insertPointer,
  const int blockDim_sh_hit,
  const int blockDim_product,
  int* const tracks_to_follow,
  int* const weak_tracks,
  Track* const tracklets,
  float* const best_fits,
  Track* tracks,
  const int number_of_hits
);

__device__ void fillCandidates(
  int* const hit_candidates,
  int* const hit_h2_candidates,
  const int number_of_sensors,
  const int* const sensor_hitStarts,
  const int* const sensor_hitNums,
  const float* const hit_Xs,
  const float* const hit_Ys,
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
  float* const sh_hit_x,
  float* const sh_hit_y,
#endif
  const float* const hit_Xs,
  const float* const hit_Ys,
  const Sensor* sensor_data,
  const int* const hit_candidates,
  unsigned int* const max_numhits_to_process,
  int* const sh_hit_process,
  bool* const hit_used,
  const int* const hit_h2_candidates,
  const int blockDim_sh_hit,
  float* const best_fits,
  unsigned int* const tracklets_insertPointer,
  unsigned int* const ttf_insertPointer,
  Track* const tracklets,
  int* const tracks_to_follow
);

__global__ void searchByTriplet(
  Track* const dev_tracks,
  const char* const dev_input,
  int* const dev_tracks_to_follow,
  bool* const dev_hit_used,
  int* const dev_atomicsStorage,
  Track* const dev_tracklets,
  int* const dev_weak_tracks,
  int* const dev_event_offsets,
  int* const dev_hit_offsets,
  float* const dev_best_fits,
  int* const dev_hit_candidates,
  int* const dev_hit_h2_candidates
);
