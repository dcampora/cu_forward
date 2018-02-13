#include "SearchByTriplet.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
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
) {
  
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const int event_number = blockIdx.x;
  const int events_under_process = gridDim.x;
  const int tracks_offset = event_number * MAX_TRACKS;

  // Pointers to data within the event
  const int data_offset = dev_event_offsets[event_number];
  const int* no_sensors = (const int*) &dev_input[data_offset];
  const int* no_hits = (const int*) (no_sensors + 1);
  const float* sensor_Zs = (const float*) (no_hits + 1);
  const int number_of_sensors = no_sensors[0];
  const int number_of_hits = no_hits[0];
  const int* sensor_hitStarts = (const int*) (sensor_Zs + number_of_sensors);
  const int* sensor_hitNums = (const int*) (sensor_hitStarts + number_of_sensors);
  const unsigned int* hit_IDs = (const unsigned int*) (sensor_hitNums + number_of_sensors);
  const float* hit_Xs = (const float*) (hit_IDs + number_of_hits);
  const float* hit_Ys = (const float*) (hit_Xs + number_of_hits);

  // Per event datatypes
  Track* tracks = dev_tracks + tracks_offset;
  unsigned int* tracks_insertPointer = (unsigned int*) dev_atomicsStorage + event_number;

  // Per side datatypes
  const int hit_offset = dev_hit_offsets[event_number];
  bool* hit_used = dev_hit_used + hit_offset;
  int* h0_candidates = dev_h0_candidates + hit_offset * 2;
  int* h2_candidates = dev_h2_candidates + hit_offset * 2;

  int* tracks_to_follow = dev_tracks_to_follow + event_number * TTF_MODULO;
  int* weak_tracks = dev_weak_tracks + hit_offset;
  Track* tracklets = dev_tracklets + hit_offset;
  unsigned int* h1_rel_indices = dev_rel_indices + event_number * MAX_NUMHITS_IN_MODULE;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  const int ip_shift = events_under_process + event_number * NUM_ATOMICS;
  unsigned int* weaktracks_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 1;
  unsigned int* tracklets_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 2;
  unsigned int* ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;
  unsigned int* sh_hit_lastPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 4;
  unsigned int* local_number_of_hits = (unsigned int*) dev_atomicsStorage + ip_shift + 5;

  __shared__ int sensor_data [9];

  // Fill candidates for both sides
  fillCandidates(
    h0_candidates,
    h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    sensor_Zs
  );

  // Process each side separately
  // A-side
  processModules(
    (Sensor*) &sensor_data[0],
    number_of_sensors-1,
    2,
    hit_used,
    h0_candidates,
    h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    sensor_Zs,
    weaktracks_insertPointer,
    tracklets_insertPointer,
    ttf_insertPointer,
    sh_hit_lastPointer,
    tracks_insertPointer,
    tracks_to_follow,
    weak_tracks,
    tracklets,
    tracks,
    number_of_hits,
    h1_rel_indices,
    local_number_of_hits
  );

  // B-side
  processModules(
    (Sensor*) &sensor_data[0],
    number_of_sensors-2,
    2,
    hit_used,
    h0_candidates,
    h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    sensor_Zs,
    weaktracks_insertPointer,
    tracklets_insertPointer,
    ttf_insertPointer,
    sh_hit_lastPointer,
    tracks_insertPointer,
    tracks_to_follow,
    weak_tracks,
    tracklets,
    tracks,
    number_of_hits,
    h1_rel_indices,
    local_number_of_hits
  );

  __syncthreads();

  // Compute the three-hit tracks left
  const unsigned int weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim.x - 1) / blockDim.x; ++i) {
    const unsigned int weaktrack_no = blockDim.x * i + threadIdx.y * blockDim.x + threadIdx.x;
    if (weaktrack_no < weaktracks_total) {
      // Load the tracks from the tracklets
      const Track t = tracklets[weak_tracks[weaktrack_no]];

      // Store them in the tracks bag iff they
      // are made out of three unused hits
      if (!hit_used[t.hits[0]] &&
          !hit_used[t.hits[1]] &&
          !hit_used[t.hits[2]]) {
        const unsigned int trackno = atomicAdd(tracks_insertPointer, 1);
        ASSERT(trackno < MAX_TRACKS)
        tracks[trackno] = t;
      }
    }
  }
}
