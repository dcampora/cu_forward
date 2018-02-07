#include "SearchByTriplet.cuh"

__device__ void processModules(
#if USE_SHARED_FOR_HITS
  float* sh_hit_x,
  float* sh_hit_y,
  float* sh_hit_z,
#endif
  int* sh_hit_process,
  int* sensor_data,
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
  const float* const hit_Zs,
  const int* sensor_Zs,
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
) {
  // Deal with odd or even in the same thread
  int first_sensor = starting_sensor;

  // Prepare s1 and s2 for the first iteration
  unsigned int prev_ttf, last_ttf = 0;

  while (first_sensor >= 4) {

    __syncthreads();
    
    // Iterate in sensors
    // Load in shared
    if (threadIdx.x < 3 && threadIdx.y == 0) {
      const int sensor_number = first_sensor - threadIdx.x * 2;
      sensor_data[threadIdx.x] = sensor_hitStarts[sensor_number];
    }
    else if (threadIdx.x >= 3 && threadIdx.x < 6 && threadIdx.y == 0) {
      const int sensor_number = first_sensor - (threadIdx.x - 3) * 2;
      sensor_data[threadIdx.x] = sensor_hitNums[sensor_number];
    }
    else if (threadIdx.x == 6 && threadIdx.y == 0) {
      sh_hit_lastPointer[0] = 0;
    }
    else if (threadIdx.x == 7 && threadIdx.y == 0) {
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
      (float*) &sh_hit_z[0],
#endif
      hit_Xs,
      hit_Ys,
      hit_Zs,
      hit_used,
      tracks_insertPointer,
      ttf_insertPointer,
      weaktracks_insertPointer,
      blockDim_sh_hit,
      (int*) &sensor_data[0],
      diff_ttf,
      blockDim_product,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      number_of_hits
    );

    // Iterate in all hits for current sensor
    // 2a. Seeding - Track creation

    // Pre-seeding 
    // Get the hits we are going to iterate onto in sh_hit_process,
    // in groups of max NUMTHREADS_X

    unsigned int sh_hit_prevPointer = 0;
    unsigned int shift_lastPointer = blockDim.x;
    while (sh_hit_prevPointer < sensor_data[SENSOR_DATA_HITNUMS]) {

      __syncthreads();
      if (threadIdx.y == 0) {
        // All threads in this context will add a hit to the 
        // shared elements, or exhaust the list
        int sh_element = sh_hit_prevPointer + threadIdx.x;
        bool inside_bounds = sh_element < sensor_data[SENSOR_DATA_HITNUMS];
        int h0_index = sensor_data[0] + sh_element;
        bool is_h0_used = inside_bounds ? hit_used[h0_index] : 1;

        // Find an unused element or exhaust the list,
        // in case the hit is used
        while (inside_bounds && is_h0_used) {
          // Since it is used, find another element while we are inside bounds
          // This is a simple gather for those elements
          sh_element = sh_hit_prevPointer + shift_lastPointer + atomicAdd(sh_hit_lastPointer, 1);
          inside_bounds = sh_element < sensor_data[SENSOR_DATA_HITNUMS];
          h0_index = sensor_data[0] + sh_element;
          is_h0_used = inside_bounds ? hit_used[h0_index] : 1;
        }

        // Fill in sh_hit_process with either the found hit or -1
        ASSERT(h0_index >= 0)
        sh_hit_process[threadIdx.x] = (inside_bounds && !is_h0_used) ? h0_index : -1;
      }
      __syncthreads();

      // Update the iteration condition
      sh_hit_prevPointer = sh_hit_lastPointer[0] + shift_lastPointer;
      shift_lastPointer += blockDim.x;

      // Track creation
      trackSeeding(
#if USE_SHARED_FOR_HITS
        (float*) &sh_hit_x[0],
        (float*) &sh_hit_y[0],
        (float*) &sh_hit_z[0],
#endif
        hit_Xs,
        hit_Ys,
        hit_Zs,
        (int*) &sensor_data[0],
        hit_candidates,
        max_numhits_to_process,
        (int*) &sh_hit_process[0],
        hit_used,
        hit_h2_candidates,
        blockDim_sh_hit,
        best_fits,
        tracklets_insertPointer,
        ttf_insertPointer,
        tracklets,
        tracks_to_follow);
    }

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
 * 
 * @details It should be simplistic in its design, as is the Pixel VELO problem ;)
 *          Triplets are chosen based on a fit and forwarded using a typical track following algo.
 *          Ghosts are inherently out of the equation, as the algorithm considers all possible
 *          triplets and keeps the best. Upon following, if a hit is not found in the adjacent
 *          module, the track[let] is considered complete.
 *          Clones are removed based off a used-hit mechanism. A global array keeps track of
 *          used hits when forming tracks consist of 4 or more hits.
 *
 *          The algorithm consists in two stages: Track following, and seeding. In each step [iteration],
 *          the track following is performed first, hits are marked as used, and then the seeding is performed,
 *          requiring the first two hits in the triplet to be unused.
 * 
 * @param dev_tracks            
 * @param dev_input             
 * @param dev_tracks_to_follow  
 * @param dev_hit_used          
 * @param dev_atomicsStorage    
 * @param dev_tracklets         
 * @param dev_weak_tracks       
 * @param dev_event_offsets     
 * @param dev_hit_offsets       
 * @param dev_best_fits         
 * @param dev_hit_candidates    
 * @param dev_hit_h2_candidates 
 */
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
) {
  
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const int event_number = blockIdx.x;
  const int events_under_process = gridDim.x;
  const int tracks_offset = event_number * MAX_TRACKS;
  const int blockDim_product = blockDim.x * blockDim.y;

  // Pointers to data within the event
  const int data_offset = dev_event_offsets[event_number];
  const int* const no_sensors = (const int*) &dev_input[data_offset];
  const int* const no_hits = (const int*) (no_sensors + 1);
  const int* const sensor_Zs = (const int*) (no_hits + 1);
  const int number_of_sensors = no_sensors[0];
  const int number_of_hits = no_hits[0];
  const int* const sensor_hitStarts = (const int*) (sensor_Zs + number_of_sensors);
  const int* const sensor_hitNums = (const int*) (sensor_hitStarts + number_of_sensors);
  const unsigned int* const hit_IDs = (const unsigned int*) (sensor_hitNums + number_of_sensors);
  const float* const hit_Xs = (const float*) (hit_IDs + number_of_hits);
  const float* const hit_Ys = (const float*) (hit_Xs + number_of_hits);
  const float* const hit_Zs = (const float*) (hit_Ys + number_of_hits);

  // Per event datatypes
  Track* tracks = dev_tracks + tracks_offset;
  unsigned int* const tracks_insertPointer = (unsigned int*) dev_atomicsStorage + event_number;

  // Per side datatypes
  const int hit_offset = dev_hit_offsets[event_number];
  bool* const hit_used = dev_hit_used + hit_offset;
  int* const hit_candidates = dev_hit_candidates + hit_offset * 2;
  int* const hit_h2_candidates = dev_hit_h2_candidates + hit_offset * 2;

  int* const tracks_to_follow = dev_tracks_to_follow + event_number * TTF_MODULO;
  int* const weak_tracks = dev_weak_tracks + hit_offset;
  Track* const tracklets = dev_tracklets + hit_offset;
  float* const best_fits = dev_best_fits + event_number * blockDim_product;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  const int ip_shift = events_under_process + event_number * NUM_ATOMICS;
  unsigned int* const weaktracks_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 1;
  unsigned int* const tracklets_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 2;
  unsigned int* const ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;
  unsigned int* const sh_hit_lastPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 4;
  unsigned int* const max_numhits_to_process = (unsigned int*) dev_atomicsStorage + ip_shift + 5;

  /* The fun begins */
#if USE_SHARED_FOR_HITS
  __shared__ float sh_hit_x [NUMTHREADS_X * SH_HIT_MULT];
  __shared__ float sh_hit_y [NUMTHREADS_X * SH_HIT_MULT];
  __shared__ float sh_hit_z [NUMTHREADS_X * SH_HIT_MULT];
#endif
  __shared__ int sh_hit_process [NUMTHREADS_X];
  __shared__ int sensor_data [6];

  const int cond_sh_hit_mult = USE_SHARED_FOR_HITS ? min(blockDim.y, SH_HIT_MULT) : blockDim.y;
  const int blockDim_sh_hit = NUMTHREADS_X * cond_sh_hit_mult;

  fillCandidates(hit_candidates,
    hit_h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    sensor_Zs);

  // Process each side separately
  processModules(
#if USE_SHARED_FOR_HITS
    (float*) &sh_hit_x[0],
    (float*) &sh_hit_y[0],
    (float*) &sh_hit_z[0],
#endif
    (int*) &sh_hit_process[0],
    (int*) &sensor_data[0],
    number_of_sensors-1,
    2,
    hit_used,
    hit_candidates,
    hit_h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
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
    number_of_hits
  );

  __syncthreads();

  processModules(
#if USE_SHARED_FOR_HITS
    (float*) &sh_hit_x[0],
    (float*) &sh_hit_y[0],
    (float*) &sh_hit_z[0],
#endif
    (int*) &sh_hit_process[0],
    (int*) &sensor_data[0],
    number_of_sensors-2,
    2,
    hit_used,
    hit_candidates,
    hit_h2_candidates,
    number_of_sensors,
    sensor_hitStarts,
    sensor_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
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
    number_of_hits
  );

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
