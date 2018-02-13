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
  Sensor* sensor_data,
  const int starting_sensor,
  const int stride,
  bool* hit_used,
  const int* h0_candidates,
  const int* h2_candidates,
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
  unsigned int* tracks_insertPointer,
  int* tracks_to_follow,
  int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  const int number_of_hits,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
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
  const float* hit_Xs,
  const float* hit_Ys,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const Sensor* sensor_data,
  const unsigned int diff_ttf,
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
  const int* h0_candidates,
  const int* h2_candidates,
  bool* hit_used,
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
  int* dev_h0_candidates,
  int* dev_h2_candidates,
  unsigned int* dev_rel_indices
);
