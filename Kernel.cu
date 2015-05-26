
#include "Kernel.cuh"

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
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float h1_z, const Hit& h2) {
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
 * @brief Fills dev_hit_candidates.
 *
 * At the moment, to evaluate its impact, it's done simply sequentially.
 * 
 * @param dev_hit_candidates 
 * @param no_sensors         
 * @param sensor_hitStarts   
 * @param sensor_hitNums     
 * @param hit_Xs             
 * @param hit_Ys             
 * @param hit_Zs             
 */
__device__ void fillCandidates(int* const hit_candidates, const int number_of_sensors,
  const int* const sensor_hitStarts, const int* const sensor_hitNums,
  const float* const hit_Xs, const float* const hit_Ys,
  const float* const hit_Zs) {

  const int blockDim_product = blockDim.x * blockDim.y;
  int first_sensor = number_of_sensors - 1;
  while (first_sensor >= 4) {
    const int second_sensor = first_sensor - 2;

    // Iterate in all hits in z0
    for (int i=0; i<((int) ceilf( ((float) sensor_hitNums[first_sensor]) / blockDim_product)); ++i) {
      const int h0_element = blockDim.x * i + threadIdx.x * blockDim.y + threadIdx.y;
      bool inside_bounds = h0_element < sensor_hitNums[first_sensor];

      if (inside_bounds) {
        bool first_found = false;
        bool last_found = false;
        const int h0_index = sensor_hitStarts[first_sensor] + h0_element;
        int h1_index;
        Hit h0 {hit_Xs[h0_index], hit_Ys[h0_index], hit_Zs[h0_index]};
        
        // Iterate in all hits in z1
        for (int h1_element=0; h1_element<sensor_hitNums[second_sensor]; ++h1_element) {
          inside_bounds = h1_element < sensor_hitNums[second_sensor];

          if (inside_bounds) {
            h1_index = sensor_hitStarts[second_sensor] + h1_element;
            Hit h1 {hit_Xs[h1_index], hit_Ys[h1_index], hit_Zs[h1_index]};

            // Check if h0 and h1 are compatible
            const float h_dist = fabs(h1.z - h0.z);
            const float dxmax = PARAM_MAXXSLOPE * h_dist;
            const bool tol_condition = fabs(h1.x - h0.x) < dxmax;
            
            // Find the first one
            if (!first_found && tol_condition) {
              hit_candidates[2 * h0_index] = h1_index;
              first_found = true;
            }
            // The last one, only if the first one has already been found
            else if (first_found && !tol_condition) {
              // hit_candidates[2 * h0_index + 1] = h1_index;
              last_found = true;
              break;
            }
          }
        }

        // Note: If first is not found, then both should be -1
        // and there wouldn't be any iteration
        if (first_found) {
          hit_candidates[2 * h0_index + 1] = h1_index + (last_found ? 0 : 1);
        }
      }
    }

    --first_sensor;
  }
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
__global__ void searchByTriplet(Track* const dev_tracks, const char* const dev_input,
  int* const dev_tracks_to_follow, bool* const dev_hit_used, int* const dev_atomicsStorage, Track* const dev_tracklets,
  int* const dev_weak_tracks, int* const dev_event_offsets, int* const dev_hit_offsets, float* const dev_best_fits,
  int* const dev_hit_candidates) {
  
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
  Track* tracks = &dev_tracks[tracks_offset];
  unsigned int* const tracks_insertPointer = (unsigned int*) dev_atomicsStorage + event_number;

  // Per side datatypes
  const int hit_offset = dev_hit_offsets[event_number];
  bool* const hit_used = dev_hit_used + hit_offset;
  int* const hit_candidates = dev_hit_candidates + hit_offset * 2;

  int* const tracks_to_follow = dev_tracks_to_follow + tracks_offset;
  int* const weak_tracks = dev_weak_tracks + tracks_offset;
  Track* const tracklets = dev_tracklets + tracks_offset;
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
  __shared__ float sh_hit_x [NUMTHREADS_X];
  __shared__ float sh_hit_y [NUMTHREADS_X];
  __shared__ float sh_hit_z [NUMTHREADS_X];
  __shared__ int sh_hit_process [NUMTHREADS_X * SH_HIT_PROCESS_MULT];
  __shared__ int sensor_data [6];

  fillCandidates(hit_candidates, number_of_sensors, sensor_hitStarts, sensor_hitNums, hit_Xs, hit_Ys, hit_Zs);

  // Deal with odd or even in the same thread
  int first_sensor = number_of_sensors - 1;

  // Prepare s1 and s2 for the first iteration
  unsigned int prev_ttf, last_ttf = 0;

  while (first_sensor >= 4) {

    __syncthreads();
    
    // Iterate in sensors
    // Load in shared
    if (threadIdx.x < 6 && threadIdx.y == 0) {
      const int sensor_number = first_sensor - (threadIdx.x % 3) * 2;
      const int* const sensor_pointer = threadIdx.x < 3 ? sensor_hitStarts : sensor_hitNums;

      sensor_data[threadIdx.x] = sensor_pointer[sensor_number];
    }

    else if (threadIdx.x == 6 && threadIdx.y == 0) {
      sh_hit_lastPointer[0] = 0;
    }

    else if (threadIdx.x == 7 && threadIdx.y == 0) {
      max_numhits_to_process[0] = 0;
    }

    __syncthreads();

    prev_ttf = last_ttf;
    last_ttf = ttf_insertPointer[0];

    // 2a. Track forwarding
    for (int i=0; i<((int) ceilf( ((float) (last_ttf - prev_ttf)) / blockDim_product)); ++i) {
      const unsigned int ttf_element = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;

      // These variables need to go here, shared memory and scope requirements
      float tx, ty, h1_z;
      unsigned int trackno, fulltrackno, skipped_modules, best_hit_h2;
      Track t;
      Hit h0;

      // The logic is broken in two parts for shared memory loading
      const bool ttf_condition = ttf_element < (last_ttf - prev_ttf);
      if (ttf_condition) {
        fulltrackno = tracks_to_follow[prev_ttf + ttf_element];
        const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
        skipped_modules = (fulltrackno & 0x70000000) >> 28;
        trackno = fulltrackno & 0x0FFFFFFF;

        const Track* const track_pointer = track_flag ? tracklets : tracks;
        t = track_pointer[trackno];

        // Load last two hits in h0, h1
        const int t_hitsNum = t.hitsNum;
        const int h0_num = t.hits[t_hitsNum - 2];
        const int h1_num = t.hits[t_hitsNum - 1];

        h0.x = hit_Xs[h0_num];
        h0.y = hit_Ys[h0_num];
        h0.z = hit_Zs[h0_num];

        const float h1_x = hit_Xs[h1_num];
        const float h1_y = hit_Ys[h1_num];
        h1_z = hit_Zs[h1_num];

        // Track forwarding over t, for all hits in the next module
        // Line calculations
        const float td = 1.0f / (h1_z - h0.z);
        const float txn = (h1_x - h0.x);
        const float tyn = (h1_y - h0.y);
        tx = txn * td;
        ty = tyn * td;
      }

      // Search for a best fit
      // Load shared elements
      
      // Iterate in the third list of hits
      // Tiled memory access on h2
      // Only load for threadIdx.y == 0
      float best_fit = MAX_FLOAT;
      for (int k=0; k<((int) ceilf( ((float) sensor_data[SENSOR_DATA_HITNUMS + 2]) / blockDim.x)); ++k) {
        
        __syncthreads();
        const int sh_hit_no = blockDim.x * k + threadIdx.x;
        if (sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2] && threadIdx.y==0) {
          const int h2_index = sensor_data[2] + sh_hit_no;

          // Coalesced memory accesses
    		  sh_hit_x[threadIdx.x] = hit_Xs[h2_index];
    		  sh_hit_y[threadIdx.x] = hit_Ys[h2_index];
    		  sh_hit_z[threadIdx.x] = hit_Zs[h2_index];
        }
        __syncthreads();

        if (ttf_condition) {
          const int last_hit_h2 = min(blockDim.x * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
          for (int kk=blockDim.x * k; kk<last_hit_h2; ++kk) {
            
            const int h2_index = sensor_data[2] + kk;
            const int sh_h2_index = kk % blockDim.x;
            const Hit h2 {sh_hit_x[sh_h2_index], sh_hit_y[sh_h2_index], sh_hit_z[sh_h2_index]};

            const float fit = fitHitToTrack(tx, ty, h0, h1_z, h2);
            const bool fit_is_better = fit < best_fit;

            best_fit = fit_is_better * fit + !fit_is_better * best_fit;
            best_hit_h2 = fit_is_better * h2_index + !fit_is_better * best_hit_h2;
          }
        }
      }

      // We have a best fit!
      // Fill in t, ONLY in case the best fit is acceptable
      if (ttf_condition) {
        if (best_fit != MAX_FLOAT) {
          // Mark h2 as used
          hit_used[best_hit_h2] = true;

          // Update the tracks to follow, we'll have to follow up
          // this track on the next iteration :)
          t.hits[t.hitsNum++] = best_hit_h2;

          // Update the track in the bag
          if (t.hitsNum <= 4) {
            // Also mark the first three as used
            hit_used[t.hits[0]] = true;
            hit_used[t.hits[1]] = true;
            hit_used[t.hits[2]] = true;

            // If it is a track made out of less than or equal than 4 hits,
            // we have to allocate it in the tracks pointer
            trackno = atomicAdd(tracks_insertPointer, 1);
          }

          // Copy the track into tracks
          tracks[trackno] = t;

          // Add the tracks to the bag of tracks to_follow
          const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
          tracks_to_follow[ttfP] = trackno;
        }
        // A track just skipped a module
        // We keep it for another round
        else if (skipped_modules <= MAX_SKIPPED_MODULES) {
          // Form the new mask
          trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

          // Add the tracks to the bag of tracks to_follow
          const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
          tracks_to_follow[ttfP] = trackno;
        }
        // If there are only three hits in this track,
        // mark it as "doubtful"
        else if (t.hitsNum == 3) {
          const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
          weak_tracks[weakP] = trackno;
        }
        // In the "else" case, we couldn't follow up the track,
        // so we won't be track following it anymore.
      }
    }

    // Iterate in all hits for current sensor
    // 2a. Seeding - Track creation

    // Pre-seeding 
    // Get the hits we are going to iterate onto in sh_hit_process,
    // in groups of max NUMTHREADS_X

    unsigned int sh_hit_prevPointer = 0;
    unsigned int shift_lastPointer = blockDim.x * SH_HIT_PROCESS_MULT;
    while (sh_hit_prevPointer < sensor_data[SENSOR_DATA_HITNUMS]) {

      __syncthreads();

      if (threadIdx.y < SH_HIT_PROCESS_MULT) {
        // All threads in this context will add a hit to the 
        // shared elements, or exhaust the list
        const int shift_sh_element = threadIdx.y * blockDim.x + threadIdx.x;
        int sh_element = sh_hit_prevPointer + shift_sh_element;
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
        sh_hit_process[shift_sh_element] = (inside_bounds && !is_h0_used) ? h0_index : -1;
        
        // TODO: Check if this is actually faster, or comment otherwise
        // sh_hit_x[shift_sh_element] = inside_bounds ? hit_Xs[h0_index] : 0;
        // sh_hit_y[shift_sh_element] = inside_bounds ? hit_Ys[h0_index] : 0;
        // sh_hit_z[shift_sh_element] = inside_bounds ? hit_Zs[h0_index] : 0;
      }

      __syncthreads();

      // Update the iteration condition
      sh_hit_prevPointer = sh_hit_lastPointer[0] + shift_lastPointer;
      shift_lastPointer += blockDim.x * SH_HIT_PROCESS_MULT;

      for (int sh_hit_iteration=0; sh_hit_iteration<SH_HIT_PROCESS_MULT; ++sh_hit_iteration) {

        // Track creation starts
        unsigned int best_hit_h1, best_hit_h2;
        Hit h0, h1, h2;
        int first_h1;
        float dymax;

        const int h0_index = sh_hit_process[threadIdx.x + sh_hit_iteration * blockDim.x];
        bool inside_bounds = h0_index != -1;
        unsigned int num_h1_to_process = 0;
        float best_fit = MAX_FLOAT;

        // We will repeat this for performance reasons
        if (inside_bounds) {
          // h0.x = sh_hit_x[threadIdx.x];
          // h0.y = sh_hit_y[threadIdx.x];
          // h0.z = sh_hit_z[threadIdx.x];
          h0.x = hit_Xs[h0_index];
          h0.y = hit_Ys[h0_index];
          h0.z = hit_Zs[h0_index];
          
          // Calculate new dymax
          const float s1_z = hit_Zs[sensor_data[1]];
          const float h_dist = fabs(s1_z - h0.z);
          dymax = PARAM_MAXYSLOPE * h_dist;

          // Only iterate in the hits indicated by hit_candidates :)
          first_h1 = hit_candidates[2 * h0_index];
          const int last_h1 = hit_candidates[2 * h0_index + 1];
          num_h1_to_process = last_h1 - first_h1;
          atomicMax(max_numhits_to_process, num_h1_to_process);
        }

        __syncthreads();

        // Only iterate max_numhits_to_process[0] iterations (with blockDim.y threads) :D :D :D
        for (int j=0; j<((int) ceilf(((float) (max_numhits_to_process[0])) / blockDim.y)); ++j) {
          const int h1_element = blockDim.y * j + threadIdx.y;
          inside_bounds &= h1_element < num_h1_to_process;
          bool is_h1_used = true; // TODO: Can be merged with h1_element restriction
          int h1_index;

          if (inside_bounds) {
            h1_index = first_h1 + h1_element;
            is_h1_used = hit_used[h1_index];
            if (!is_h1_used) {
              h1.x = hit_Xs[h1_index];
              h1.y = hit_Ys[h1_index];
              h1.z = hit_Zs[h1_index];
            }
          }

          // Iterate in the third list of hits
          // Tiled memory access on h2
          for (int k=0; k<((int) ceilf( ((float) sensor_data[SENSOR_DATA_HITNUMS + 2]) / blockDim.x)); ++k) {

            __syncthreads();
            const int sh_hit_no = blockDim.x * k + threadIdx.x;
            if (sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2] && threadIdx.y==0) {
              const int h2_index = sensor_data[2] + sh_hit_no;

              // Coalesced memory accesses
              sh_hit_x[threadIdx.x] = hit_Xs[h2_index];
        			sh_hit_y[threadIdx.x] = hit_Ys[h2_index];
        			sh_hit_z[threadIdx.x] = hit_Zs[h2_index];
            }
            __syncthreads();

            if (inside_bounds && !is_h1_used) {

              const int last_hit_h2 = min(blockDim.x * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
              for (int kk=blockDim.x * k; kk<last_hit_h2; ++kk) {

                const int h2_index = sensor_data[2] + kk;
                const int sh_h2_index = kk % blockDim.x;
                h2.x = sh_hit_x[sh_h2_index];
                h2.y = sh_hit_y[sh_h2_index];
                h2.z = sh_hit_z[sh_h2_index];

                // Predictions of x and y for this hit
                const float z2_tz = (h2.z - h0.z) / (h1.z - h0.z);
                const float x = h0.x + (h1.x - h0.x) * z2_tz;
                const float y = h0.y + (h1.y - h0.y) * z2_tz;
                const float dx = x - h2.x;
                const float dy = y - h2.y;

                if (fabs(h1.y - h0.y) < dymax && fabs(dx) < PARAM_TOLERANCE_EXTRA && fabs(dy) < PARAM_TOLERANCE_EXTRA) {
                  // Calculate fit
                  const float scatterNum = (dx * dx) + (dy * dy);
                  const float scatterDenom = 1.f / (h2.z - h1.z);
                  const float scatter = scatterNum * scatterDenom * scatterDenom;
                  const bool condition = scatter < MAX_SCATTER;
                  const float fit = condition * scatter + !condition * MAX_FLOAT; 

                  const bool fit_is_better = fit < best_fit;
                  best_fit = fit_is_better * fit + !fit_is_better * best_fit;
                  best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
                  best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
                }
              }
            }
          }
        }

        // Compare / Mix the results from the blockDim.y threads
        best_fits[threadIdx.x * blockDim.y + threadIdx.y] = best_fit;

        __syncthreads();

        bool accept_track = false;
        if (h0_index != -1 && best_fit != MAX_FLOAT) {
          best_fit = MAX_FLOAT;
          int threadIdx_y_winner = -1;
          for (int i=0; i<blockDim.y; ++i) {
            const float fit = best_fits[threadIdx.x * blockDim.y + i];
            if (fit < best_fit) {
              best_fit = fit;
              threadIdx_y_winner = i;
            }
          }
          accept_track = threadIdx.y == threadIdx_y_winner;
        }

        // We have a best fit! - haven't we?
        // Only go through the tracks on the selected thread
        if (accept_track) {
          // Fill in track information

          // Add the track to the bag of tracks
          const unsigned int trackP = atomicAdd(tracklets_insertPointer, 1);
          tracklets[trackP] = Track {3, (unsigned int) sh_hit_process[threadIdx.x + sh_hit_iteration * blockDim.x], best_hit_h1, best_hit_h2};;

          // Add the tracks to the bag of tracks to_follow
          // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
          // and hence it is stored in tracklets
          const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
          tracks_to_follow[ttfP] = 0x80000000 | trackP;
        }
      }
    }

    first_sensor -= 1;
  }

  __syncthreads();

  prev_ttf = last_ttf;
  last_ttf = ttf_insertPointer[0];

  // Process the last bunch of track_to_follows
  for (int i=0; i<((int) ceilf( ((float) (last_ttf - prev_ttf)) / blockDim_product)); ++i) {
    const unsigned int ttf_element = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;

    if (ttf_element < (last_ttf - prev_ttf)) {
      const int fulltrackno = tracks_to_follow[prev_ttf + ttf_element];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

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
  for (int i=0; i<((int) ceilf( ((float) weaktracks_total) / blockDim_product)); ++i) {
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
        tracks[trackno] = t;
      }
    }
  }
}
