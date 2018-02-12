#include "SearchByTriplet.cuh"

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
) {
  // Deal with odd or even in the same thread
  int first_sensor = starting_sensor;

  // Prepare s1 and s2 for the first iteration
  unsigned int prev_ttf, last_ttf = 0;

  while (first_sensor >= 4) {

    __syncthreads();
    
    // Iterate in sensors
    // Load in shared
    if (threadIdx.x < 3) {
      const int sensor_number = first_sensor - threadIdx.x * 2;
      sensor_data[threadIdx.x].hitStart = sensor_hitStarts[sensor_number];
      sensor_data[threadIdx.x].hitNums = sensor_hitNums[sensor_number];
      sensor_data[threadIdx.x].z = sensor_Zs[sensor_number];
    }
    else if (threadIdx.x == 4) {
      sh_hit_lastPointer[0] = 0;
    }
    else if (threadIdx.x == 5) {
      max_numhits_to_process[0] = 0;
    }

    prev_ttf = last_ttf;
    last_ttf = ttf_insertPointer[0];
    const unsigned int diff_ttf = last_ttf - prev_ttf;

    __syncthreads();

    // 2a. Track forwarding
    trackForwarding(
#if USE_SHARED_FOR_HITS
      (float*) &sh_hit_x[0],
      (float*) &sh_hit_y[0],
#endif
      hit_Xs,
      hit_Ys,
      hit_used,
      tracks_insertPointer,
      ttf_insertPointer,
      weaktracks_insertPointer,
      blockDim_sh_hit,
      sensor_data,
      diff_ttf,
      blockDim_product,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      number_of_hits,
      first_sensor,
      sensor_Zs,
      sensor_hitStarts,
      sensor_hitNums
    );

    __syncthreads();

    // Seeding
    trackSeeding(
      hit_Xs,
      hit_Ys,
      sensor_data,
      hit_candidates,
      max_numhits_to_process,
      hit_used,
      hit_h2_candidates,
      blockDim_sh_hit,
      best_fits,
      tracklets_insertPointer,
      ttf_insertPointer,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      local_number_of_hits
    );

    first_sensor -= stride;
  }

  __syncthreads();

  prev_ttf = last_ttf;
  last_ttf = ttf_insertPointer[0];
  const unsigned int diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (int i=0; i<(diff_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;

    if (ttf_element < diff_ttf) {
      const int fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % TTF_MODULO];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
        ASSERT(weakP < number_of_hits)
        weak_tracks[weakP] = trackno;
      }
    }
  }
}

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
  float* dev_best_fits,
  int* dev_hit_candidates,
  int* dev_hit_h2_candidates,
  unsigned int* dev_rel_indices
) {
  
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const int event_number = blockIdx.x;
  const int events_under_process = gridDim.x;
  const int tracks_offset = event_number * MAX_TRACKS;
  const int blockDim_product = blockDim.x * blockDim.y;

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
  int* hit_candidates = dev_hit_candidates + hit_offset * 2;
  int* hit_h2_candidates = dev_hit_h2_candidates + hit_offset * 2;

  int* tracks_to_follow = dev_tracks_to_follow + event_number * TTF_MODULO;
  int* weak_tracks = dev_weak_tracks + hit_offset;
  Track* tracklets = dev_tracklets + hit_offset;
  float* best_fits = dev_best_fits + event_number * blockDim_product;
  unsigned int* h1_rel_indices = dev_rel_indices + event_number * MAX_NUMHITS_IN_MODULE;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  const int ip_shift = events_under_process + event_number * NUM_ATOMICS;
  unsigned int* weaktracks_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 1;
  unsigned int* tracklets_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 2;
  unsigned int* ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;
  unsigned int* sh_hit_lastPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 4;
  unsigned int* max_numhits_to_process = (unsigned int*) dev_atomicsStorage + ip_shift + 5;
  unsigned int* local_number_of_hits = (unsigned int*) dev_atomicsStorage + ip_shift + 6;

  /* The fun begins */
#if USE_SHARED_FOR_HITS
  __shared__ float sh_hit_x [NUMTHREADS_X];
  __shared__ float sh_hit_y [NUMTHREADS_X];
#endif
  __shared__ int sensor_data [9];

  const int blockDim_sh_hit = NUMTHREADS_X;

  fillCandidates(hit_candidates,
    hit_h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    sensor_Zs);

  // Process each side separately
  processModules(
#if USE_SHARED_FOR_HITS
    (float*) &sh_hit_x[0],
    (float*) &sh_hit_y[0],
#endif
    (Sensor*) &sensor_data[0],
    number_of_sensors-1,
    1,
    hit_used,
    hit_candidates,
    hit_h2_candidates,
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
    max_numhits_to_process,
    tracks_insertPointer,
    blockDim_sh_hit,
    blockDim_product,
    tracks_to_follow,
    weak_tracks,
    tracklets,
    best_fits,
    tracks,
    number_of_hits,
    h1_rel_indices,
    local_number_of_hits
  );

//   __syncthreads();

//   processModules(
// #if USE_SHARED_FOR_HITS
//     (float*) &sh_hit_x[0],
//     (float*) &sh_hit_y[0],
// #endif
//     (int*) &sh_hit_process[0],
//     (int*) &sensor_data[0],
//     number_of_sensors-2,
//     2,
//     hit_used,
//     hit_candidates,
//     hit_h2_candidates,
//     number_of_sensors,
//     sensor_hitStarts,
//     sensor_hitNums,
//     hit_Xs,
//     hit_Ys,
//     sensor_Zs,
//     weaktracks_insertPointer,
//     tracklets_insertPointer,
//     ttf_insertPointer,
//     sh_hit_lastPointer,
//     max_numhits_to_process,
//     tracks_insertPointer,
//     blockDim_sh_hit,
//     blockDim_product,
//     tracks_to_follow,
//     weak_tracks,
//     tracklets,
//     best_fits,
//     tracks,
//     number_of_hits
//   );

  __syncthreads();

  // Compute the three-hit tracks left
  const unsigned int weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int weaktrack_no = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;
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
