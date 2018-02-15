#include "SearchByTriplet.cuh"

/**
 * @brief Fits hits to tracks.
 * 
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns FLT_MAX.
 */
__device__ float fitHitToTrack(
  const float tx,
  const float ty,
  const Hit& h0,
  const float h0_z,
  const float h1_z,
  const Hit& h2,
  const float h2_z
) {
  // tolerances
  const float dz = h2_z - h0_z;
  const float x_prediction = h0.x + tx * dz;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < PARAM_TOLERANCE;

  const float y_prediction = h0.y + ty * dz;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < PARAM_TOLERANCE;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2_z - h1_z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool scatter_condition = scatter < MAX_SCATTER_FORWARDING;
  const bool condition = tolx_condition && toly_condition && scatter_condition;

  return condition * scatter + !condition * FLT_MAX;
}

/**
 * @brief Performs the track forwarding of forming tracks
 */
__device__ void trackForwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const Module* module_data,
  const unsigned int diff_ttf,
  int* tracks_to_follow,
  int* weak_tracks,
  const unsigned int prev_ttf,
  Track* tracklets,
  Track* tracks,
  const int number_of_hits,
  const int first_module,
  const float* module_Zs,
  const int* module_hitStarts,
  const int* module_hitNums
) {
  for (int i=0; i<(diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const unsigned int ttf_element = blockDim.x * i + threadIdx.y * blockDim.x + threadIdx.x;

    // These variables need to go here, shared memory and scope requirements
    float tx, ty, h1_z, h0_z;
    unsigned int trackno, fulltrackno, skipped_modules, best_hit_h2;
    Track t;
    Hit h0;

    // The logic is broken in two parts for shared memory loading
    const bool ttf_condition = ttf_element < diff_ttf;
    if (ttf_condition) {
      fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % TTF_MODULO];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      skipped_modules = (fulltrackno & 0x70000000) >> 28;
      trackno = fulltrackno & 0x0FFFFFFF;

      const Track* track_pointer = track_flag ? tracklets : tracks;
      
      ASSERT(track_pointer==tracklets ? trackno < number_of_hits : true)
      ASSERT(track_pointer==tracks ? trackno < MAX_TRACKS : true)
      t = track_pointer[trackno];

      // Load last two hits in h0, h1
      const int t_hitsNum = t.hitsNum;
      ASSERT(t_hitsNum < MAX_TRACK_SIZE)
      const int h0_num = t.hits[t_hitsNum - 2];
      const int h1_num = t.hits[t_hitsNum - 1];

      ASSERT(h0_num < number_of_hits)
      h0.x = hit_Xs[h0_num];
      h0.y = hit_Ys[h0_num];

      ASSERT(h1_num < number_of_hits)
      const float h1_x = hit_Xs[h1_num];
      const float h1_y = hit_Ys[h1_num];

      // 99% of the times the last two hits came from consecutive modules
      if (h0_num < module_data[0].hitStart + module_data[0].hitNums) {
        h0_z = module_data[0].z;
        h1_z = module_data[1].z;
      } else {
        // Oh boy.
        // We assume only one module can be skipped
        h1_z = (h1_num < module_data[1].hitStart + module_data[1].hitNums) ? module_data[1].z : module_data[0].z;

        // We do not know if h0 was in the previous module or the previous-previous one.
        // So we have to pay the price and ask the question
        if (h0_num < module_hitStarts[first_module+2] + module_hitNums[first_module+2]) {
          h0_z = module_Zs[first_module+2];
        } else {
          h0_z = module_Zs[first_module+4];
        }
      }

      // Track forwarding over t, for all hits in the next module
      // Line calculations
      const float td = 1.0f / (h1_z - h0_z);
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
    float best_fit = FLT_MAX;
    for (int k=0; k<(module_data[2].hitNums + blockDim.x - 1) / blockDim.x; ++k) {
      
#if USE_SHARED_FOR_HITS
      __syncthreads();
      const int shared_hit_no = blockDim.x * k + threadIdx.x;
      if (shared_hit_no < module_data[2].hitNums) {
        const int h2_index = module_data[2].hitStart + shared_hit_no;

        // Coalesced memory accesses
        ASSERT(tid < blockDim.x)
        shared_hit_x[tid] = hit_Xs[h2_index];
        shared_hit_y[tid] = hit_Ys[h2_index];
      }
      __syncthreads();
#endif
      
      if (ttf_condition) {
        const int last_hit_h2 = min(blockDim.x * (k + 1), module_data[2].hitNums);
        for (int kk=blockDim.x * k; kk<last_hit_h2; ++kk) {
          
          const int h2_index = module_data[2].hitStart + kk;
#if USE_SHARED_FOR_HITS
          const int shared_h2_index = kk % blockDim.x;
          const Hit h2 {shared_hit_x[shared_h2_index], shared_hit_y[shared_h2_index]};
#else
          const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index]};
#endif

          const float fit = fitHitToTrack(tx, ty, h0, h0_z, h1_z, h2, module_data[2].z);
          const bool fit_is_better = fit < best_fit;

          best_fit = fit_is_better ? fit : best_fit;
          best_hit_h2 = fit_is_better ? h2_index : best_hit_h2;
        }
      }
    }

    // We have a best fit!
    // Fill in t, ONLY in case the best fit is acceptable
    if (ttf_condition) {
      if (best_fit != FLT_MAX) {
        // Mark h2 as used
        ASSERT(best_hit_h2 < number_of_hits)
        hit_used[best_hit_h2] = true;

        // Update the tracks to follow, we'll have to follow up
        // this track on the next iteration :)
        ASSERT(t.hitsNum < MAX_TRACK_SIZE)
        t.hits[t.hitsNum++] = best_hit_h2;

        // Update the track in the bag
        if (t.hitsNum <= 4) {
          ASSERT(t.hits[0] < number_of_hits)
          ASSERT(t.hits[1] < number_of_hits)
          ASSERT(t.hits[2] < number_of_hits)

          // Also mark the first three as used
          hit_used[t.hits[0]] = true;
          hit_used[t.hits[1]] = true;
          hit_used[t.hits[2]] = true;

          // If it is a track made out of less than or equal than 4 hits,
          // we have to allocate it in the tracks pointer
          trackno = atomicAdd(tracks_insertPointer, 1);
        }

        // Copy the track into tracks
        ASSERT(trackno < number_of_hits)
        tracks[trackno] = t;

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // A track just skipped a module
      // We keep it for another round
      else if (skipped_modules <= MAX_SKIPPED_MODULES) {
        // Form the new mask
        trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // If there are only three hits in this track,
      // mark it as "doubtful"
      else if (t.hitsNum == 3) {
        const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
        ASSERT(weakP < number_of_hits)
        weak_tracks[weakP] = trackno;
      }
      // In the "else" case, we couldn't follow up the track,
      // so we won't be track following it anymore.
    }
  }
}
