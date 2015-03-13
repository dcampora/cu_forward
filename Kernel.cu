
#include "Kernel.cuh"

/**
 * @brief Gives the fit between h0, h1 and h2
 * @details The result is given in a float. MAX_FLOAT is
 *          used as an upper limit. The lower it is, the better the fit is.
 *          
 * @param  h0 
 * @param  h1 
 * @param  h2 
 * @return    
 */
__device__ float fitHits(const Hit& h0, const Hit& h1, const Hit &h2, const float dxmax, const float dymax) {
  // Max dx, dy permissible over next hit

  // First approximation -
  // With the sensor z, instead of the hit z
  const float z2_tz = (h2.z - h0.z) / (h1.z - h0.z);
  const float x = h0.x + (h1.x - h0.x) * z2_tz;
  const float y = h0.y + (h1.y - h0.y) * z2_tz;

  const float dx = x - h2.x;
  const float dy = y - h2.y;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2.z - h1.z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool scatter_condition = scatter < MAX_SCATTER;
  const bool condition = fabs(h1.x - h0.x) < dxmax && fabs(h1.y - h0.y) < dymax && scatter_condition;

  return condition * scatter + !condition * MAX_FLOAT;
}

/**
 * @brief Fits hits to tracks.
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns MAX_FLOAT.
 * 
 * @param tx 
 * @param ty 
 * @param h0 
 * @param h1_z
 * @param h2 
 * @return 
 */
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float h1_z, const Hit& h2){
  // tolerances
  const float dz = h2.z - h0.z;
  const float x_prediction = h0.x + tx * dz;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < PARAM_TOLERANCE;

  const float y_prediction = h0.y + ty * dz;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < PARAM_TOLERANCE;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2.z - h1_z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool scatter_condition = scatter < MAX_SCATTER;
  const bool condition = tolx_condition && toly_condition && scatter_condition;

  return condition * scatter + !condition * MAX_FLOAT;
}

/**
 * @brief Track following algorithm, loosely based on Pr/PrPixel.
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
 * @param dev_tracks_to_follow_q1 
 * @param dev_tracks_to_follow_q2 
 * @param dev_hit_used            
 * @param dev_atomicsStorage      
 * @param dev_tracklets           
 * @param dev_weak_tracks         
 * @param dev_event_offsets       
 * @param dev_hit_offsets         
 */

__global__ void sbt_seeding (const char* const dev_input,
  int* const dev_tracks_to_follow, int* const dev_atomicsStorage,
  Track* const dev_tracklets, int* const dev_weak_tracks,
  int* const dev_event_offsets, int* const dev_hit_offsets,
  int* const dev_ttf_per_module) {

  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const int num_modules = 52;
  const int event_number = blockIdx.x;
  const int sensor_side = blockIdx.y;
  const int events_under_process = gridDim.x;
  const int tracks_sides_offset = 2 * event_number * MAX_TRACKS + sensor_side * MAX_TRACKS;

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
  int* const ttf_per_module = dev_ttf_per_module + event_number * num_modules;
  Track* const tracklets = dev_tracklets + tracks_sides_offset;
  int* const tracks_to_follow = dev_tracks_to_follow + tracks_sides_offset;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  // Note: Adjusted to be the same as in the forwarding stage :)
  const int insertPointer_num = 4;
  const int ip_shift = events_under_process + event_number * insertPointer_num * 2 + insertPointer_num * sensor_side;
  unsigned int* const tracklets_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 2;
  unsigned int* const ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;

  /* The fun begins */
  Track t;
  Sensor s0, s1, s2;
  Hit h0, h1, h2;
  int best_hit_h1, best_hit_h2;
  
  __shared__ float sh_hit_x [64];
  __shared__ float sh_hit_y [64];
  __shared__ float sh_hit_z [64];

  // Deal with odd or even separately
  int first_sensor = num_modules - sensor_side - 1;

  // Prepare s1 and s2 for the first iteration
  const int second_sensor = first_sensor - 2;

  s1.hitStart = sensor_hitStarts[first_sensor];
  s1.hitNums = sensor_hitNums[first_sensor];
  s2.hitStart = sensor_hitStarts[second_sensor];
  s2.hitNums = sensor_hitNums[second_sensor];

  while (first_sensor >= 4) {
    // Iterate in sensors
    // Reuse the info from last sensors
    s0 = s1;
    s1 = s2;

    const int third_sensor = first_sensor - 4;
    s2.hitStart = sensor_hitStarts[third_sensor];
    s2.hitNums = sensor_hitNums[third_sensor];

    // Iterate in all hits for current sensor
    // 2a. Seeding - Track creation
    for (int i=0; i<((int) ceilf( ((float) s0.hitNums) / blockDim.x)); ++i) {
      const int first_hit = blockDim.x * i + threadIdx.x;
      const int h0_index = s0.hitStart + first_hit;
      const bool inside_bounds = first_hit < s0.hitNums;
      float best_fit = MAX_FLOAT;

      // We repeat this here for performance reasons
      if (inside_bounds){
          h0.x = hit_Xs[h0_index];
          h0.y = hit_Ys[h0_index];
          h0.z = hit_Zs[h0_index];
      }

      for (int j=0; j<s1.hitNums; ++j) {
        float dxmax, dymax;

        const int h1_index = s1.hitStart + j;
        if (inside_bounds){
          h1.x = hit_Xs[h1_index];
          h1.y = hit_Ys[h1_index];
          h1.z = hit_Zs[h1_index];

          const float h_dist = fabs((float) ( h1.z - h0.z ));
          dxmax = PARAM_MAXXSLOPE * h_dist;
          dymax = PARAM_MAXYSLOPE * h_dist;
        }

        // Iterate in the third list of hits
        // Tiled memory access on h2
        for (int k=0; k<((int) ceilf( ((float) s2.hitNums) / blockDim.x)); ++k){
          
          __syncthreads();
          const int sh_hit_no = blockDim.x * k + threadIdx.x;
          if (sh_hit_no < s2.hitNums){
            const int h2_index = s2.hitStart + sh_hit_no;

            // Coalesced memory accesses
            sh_hit_x[threadIdx.x] = hit_Xs[h2_index];
            sh_hit_y[threadIdx.x] = hit_Ys[h2_index];
            sh_hit_z[threadIdx.x] = hit_Zs[h2_index];
          }
          __syncthreads();

          if (inside_bounds){

            const int last_hit_h2 = min(blockDim.x * k + blockDim.x, s2.hitNums);
            for (int kk=blockDim.x * k; kk<last_hit_h2; ++kk){
              
              const int h2_index = s2.hitStart + kk;
              const int sh_h2_index = kk % blockDim.x;
              h2.x = sh_hit_x[sh_h2_index];
              h2.y = sh_hit_y[sh_h2_index];
              h2.z = sh_hit_z[sh_h2_index];

              const float fit = fitHits(h0, h1, h2, dxmax, dymax);
              const bool fit_is_better = fit < best_fit;

              best_fit = fit_is_better * fit + !fit_is_better * best_fit;
              best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
              best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
            }
          }
        }
      }

      // We have a best fit! - haven't we?
      const bool accept_track = best_fit != MAX_FLOAT;
      if (accept_track) {
        // Reload h1 and h2
        h1.x = hit_Xs[best_hit_h1];
        h1.y = hit_Ys[best_hit_h1];
        h1.z = hit_Zs[best_hit_h1];

        h2.x = hit_Xs[best_hit_h2];
        h2.y = hit_Ys[best_hit_h2];
        h2.z = hit_Zs[best_hit_h2];

        // Fill in track information
        t.hitsNum = 3;
        t.hits[0] = s0.hitStart + first_hit;
        t.hits[1] = best_hit_h1;
        t.hits[2] = best_hit_h2;

        // Add the track to the bag of tracks
        const unsigned int trackP = atomicAdd(tracklets_insertPointer, 1);
        tracklets[trackP] = t;

        // Add the tracks to the bag of tracks to_follow
        // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
        // and hence it is stored in tracklets
        const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
        tracks_to_follow[ttfP] = 0x80000000 | trackP;
      }
    }

    // Get the latest ttf_inserPointer
    __syncthreads();

    if (threadIdx.x == 0)
      ttf_per_module[first_sensor] = ttf_insertPointer[0];

    __syncthreads(); // I don't think this is needed

    first_sensor -= 2;
  }
}


__global__ void sbt_forwarding(const char* const dev_input, Track* const dev_tracks,
  int* const dev_tracks_to_follow, bool* const dev_hit_used, int* const dev_atomicsStorage,
  Track* const dev_tracklets, int* const dev_weak_tracks, int* const dev_event_offsets,
  int* const dev_hit_offsets, int* const dev_ttf_per_module) {
  
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const int num_modules = 52;
  const int event_number = blockIdx.x;
  const int sensor_side = blockIdx.y;
  const int events_under_process = gridDim.x;

  const int tracks_offset = event_number * MAX_TRACKS;
  const int tracks_sides_offset = 2 * event_number * MAX_TRACKS + sensor_side * MAX_TRACKS;

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
  Track* tracks = &dev_tracks[tracks_offset];
  unsigned int* const tracks_insertPointer = (unsigned int*) dev_atomicsStorage + event_number;

  // Per side datatypes
  const int hit_offset = dev_hit_offsets[event_number];
  bool* const hit_used = dev_hit_used + hit_offset;

  int* const ttf_per_module = dev_ttf_per_module + event_number * num_modules;
  int* const tracks_to_follow = dev_tracks_to_follow + tracks_sides_offset;
  int* const weak_tracks = dev_weak_tracks + tracks_sides_offset;
  Track* const tracklets = dev_tracklets + tracks_sides_offset;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  const int insertPointer_num = 4;
  const int ip_shift = events_under_process + event_number * insertPointer_num * 2 + insertPointer_num * sensor_side;
  unsigned int* const weaktracks_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 1;
  unsigned int* ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;

  /* The fun begins */
  Track t;
  Sensor s2;
  Hit h0, h1, h2;
  int best_hit_h2;
  
  __shared__ float sh_hit_x [64];
  __shared__ float sh_hit_y [64];
  __shared__ float sh_hit_z [64];

  // Deal with odd or even separately
  int first_sensor = num_modules - sensor_side - 1;
  unsigned int last_ttf = ttf_insertPointer[0];

  while (first_sensor >= 4) {
    // Iterate in sensors

    const int third_sensor = first_sensor - 4;
    s2.hitStart = sensor_hitStarts[third_sensor];
    s2.hitNums = sensor_hitNums[third_sensor];

    // Tracks to follow from seeding stage
    const unsigned int prev_seedttf = (first_sensor == num_modules-1) ? 0 : ttf_per_module[first_sensor+1];
    const unsigned int last_seedttf = ttf_per_module[first_sensor];

    assert(first_sensor < num_modules);
    assert(first_sensor > 0);

    assert(prev_seedttf < ttf_insertPointer[0]);
    assert(last_seedttf < ttf_insertPointer[0] && "last seed failed on sensor " + to_string(first_sensor) + ", last seed " + to_string(last_seedttf));

    // New ttfs
    __syncthreads();
    const unsigned int prev_ttf = last_ttf;
    last_ttf = ttf_insertPointer[0];
    // __syncthreads(); // Probably we don't need this one

    // 2a. Track following
    const unsigned int total_ttfs = (last_seedttf - prev_seedttf) + (last_ttf - prev_ttf);
    for (int i=0; i<((int) ceilf( ((float) total_ttfs) / blockDim.x)); ++i) {
      const unsigned int ttf_element = blockDim.x * i + threadIdx.x;
      const unsigned int ttf_padding = (ttf_element < last_seedttf - prev_seedttf) ? prev_seedttf : prev_ttf;

      // These variables need to go here, shared memory and scope requirements
      float tx, ty;
      int trackno;
      bool track_flag;

      // The logic is broken in two parts for shared memory loading
      bool ttf_condition = ttf_element < total_ttfs;
      if (ttf_condition) {
        const int fulltrackno = tracks_to_follow[ttf_padding + ttf_element];
        track_flag = (fulltrackno & 0x80000000) == 0x80000000;
        trackno = fulltrackno & 0x7FFFFFFF;

        const Track* track_pointer = track_flag ? tracklets : tracks;
        t = track_pointer[trackno];

        // Load last two hits in h0, h1
        const int t_hitsNum = t.hitsNum;
        const int h0_num = t.hits[t_hitsNum - 2];
        const int h1_num = t.hits[t_hitsNum - 1];

        const bool h0_used = hit_used[h0_num];
        const bool h1_used = hit_used[h1_num];

        // Update the condition with whether h0 and h1 are not used
        ttf_condition &= !h0_used && !h1_used;
        // ttf_condition = !h0_used && !h1_used; // &= is not needed here

        if (ttf_condition) {
          h0.x = hit_Xs[h0_num];
          h0.y = hit_Ys[h0_num];
          h0.z = hit_Zs[h0_num];

          h1.x = hit_Xs[h1_num];
          h1.y = hit_Ys[h1_num];
          h1.z = hit_Zs[h1_num];

          // Track following over t, for all hits in the next module
          // Line calculations
          const float td = 1.0f / (h1.z - h0.z);
          const float txn = (h1.x - h0.x);
          const float tyn = (h1.y - h0.y);
          tx = txn * td;
          ty = tyn * td;
        }
      }

      // Search for a best fit
      // Load shared elements
      
      // Iterate in the third list of hits
      // Tiled memory access on h2
      float best_fit = MAX_FLOAT;
      for (int k=0; k<((int) ceilf( ((float) s2.hitNums) / blockDim.x)); ++k){
        
        __syncthreads();
        const int sh_hit_no = blockDim.x * k + threadIdx.x;
        if (sh_hit_no < s2.hitNums){
          const int h2_index = s2.hitStart + sh_hit_no;

          // Coalesced memory accesses
          sh_hit_x[threadIdx.x] = hit_Xs[h2_index];
          sh_hit_y[threadIdx.x] = hit_Ys[h2_index];
          sh_hit_z[threadIdx.x] = hit_Zs[h2_index];
        }
        __syncthreads();

        if (ttf_condition){
          const int last_hit_h2 = min(blockDim.x * k + blockDim.x, s2.hitNums);
          for (int kk=blockDim.x * k; kk<last_hit_h2; ++kk){
            
            const int h2_index = s2.hitStart + kk;
            const int sh_h2_index = kk % blockDim.x;
            h2.x = sh_hit_x[sh_h2_index];
            h2.y = sh_hit_y[sh_h2_index];
            h2.z = sh_hit_z[sh_h2_index];

            const float fit = fitHitToTrack(tx, ty, h0, h1.z, h2);
            const bool fit_is_better = fit < best_fit;

            best_fit = fit_is_better * fit + !fit_is_better * best_fit;
            best_hit_h2 = fit_is_better * h2_index + !fit_is_better * best_hit_h2;
          }
        }
      }

      // We have a best fit!
      // Fill in t, ONLY in case the best fit is acceptable
      if (ttf_condition) {
        // if (best_fit != MAX_FLOAT) {
        //   // Reload h2
        //   h2.x = hit_Xs[best_hit_h2];
        //   h2.y = hit_Ys[best_hit_h2];
        //   h2.z = hit_Zs[best_hit_h2];

        //   // Mark h2 as used
        //   hit_used[best_hit_h2] = true;
          
        //   // Update the tracks to follow, we'll have to follow up
        //   // this track on the next iteration :)
        //   t.hits[t.hitsNum++] = best_hit_h2;

        //   // Update the track in the bag
        //   if (t.hitsNum > 4){
        //     // If it is a track made out of *strictly* more than four hits,
        //     // the trackno refers to the tracks location.
        //     tracks[trackno] = t;
        //   }
        //   else {
        //     // Otherwise, we have to allocate it in the tracks,
        //     // and update trackno
        //     trackno = atomicAdd(tracks_insertPointer, 1);
        //     tracks[trackno] = t;

        //     // Also mark the first three as used
        //     hit_used[t.hits[0]] = true;
        //     hit_used[t.hits[1]] = true;
        //     hit_used[t.hits[2]] = true;
        //   }

        //   // Add the tracks to the bag of tracks to_follow
        //   const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
        //   tracks_to_follow[ttfP] = trackno;
        // }
        // // In the "else" case, we couldn't follow up the track,
        // // so we won't be track following it anymore.
        // else if (track_flag){
          // If there are only three hits in this track,
          // mark it as "doubtful"
          const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
          weak_tracks[weakP] = trackno;
        // }
      }
    }

    first_sensor -= 2;
  }

  __syncthreads();

  // Process the last bunch of track_to_follows
  const unsigned int last_ttf_insertPointer = ttf_insertPointer[0];
  for (int i=0; i<((int) ceilf( ((float) last_ttf_insertPointer) / blockDim.x)); ++i) {
    const unsigned int ttf_element = blockDim.x * i + threadIdx.x;

    if (ttf_element < last_ttf_insertPointer) {
      const int fulltrackno = tracks_to_follow[ttf_element];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x7FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
        weak_tracks[weakP] = trackno;
      }
    }
  }

  __syncthreads();

  // Compute the three-hit tracks left
  const unsigned int weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<((int) ceilf( ((float) weaktracks_total) / blockDim.x)); ++i) {
    const unsigned int weaktrack_no = blockDim.x * i + threadIdx.x;
    if (weaktrack_no < weaktracks_total){
      // Load the tracks from the tracklets
      t = tracklets[weak_tracks[weaktrack_no]];

      // Store them in the tracks bag iff they
      // are made out of three unused hits
      if (!hit_used[t.hits[0]] &&
          !hit_used[t.hits[1]] &&
          !hit_used[t.hits[2]]){
        const unsigned int trackno = atomicAdd(tracks_insertPointer, 1);
        tracks[trackno] = t;
      }
    }
  }
}
