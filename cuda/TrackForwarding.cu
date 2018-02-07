#include "SearchByTriplet.cuh"

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
__device__ float fitHitToTrack(
  const float tx,
  const float ty,
  const Hit& h0,
  const float h1_z,
  const Hit& h2
) {
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
 * @brief Performs the track forwarding.
 *
 * @param hit_Xs           
 * @param hit_Ys           
 * @param hit_Zs           
 * @param sensor_data      
 * @param sh_hit_x         
 * @param sh_hit_y         
 * @param sh_hit_z         
 * @param diff_ttf         
 * @param blockDim_product 
 * @param tracks_to_follow 
 * @param weak_tracks      
 * @param prev_ttf         
 * @param tracklets        
 * @param tracks           
 * @param number_of_hits   
 */
__device__ void trackForwarding(
#if USE_SHARED_FOR_HITS
  float* const sh_hit_x,
  float* const sh_hit_y,
  float* const sh_hit_z,
#endif
  const float* const hit_Xs,
  const float* const hit_Ys,
  const float* const hit_Zs,
  bool* const hit_used,
  unsigned int* const tracks_insertPointer,
  unsigned int* const ttf_insertPointer,
  unsigned int* const weaktracks_insertPointer,
  const int blockDim_sh_hit,
  int* const sensor_data,
  const unsigned int diff_ttf,
  const int blockDim_product,
  int* const tracks_to_follow,
  int* const weak_tracks,
  const unsigned int prev_ttf,
  Track* const tracklets,
  Track* const tracks,
  const int number_of_hits
) {
  for (int i=0; i<(diff_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;

    // These variables need to go here, shared memory and scope requirements
    float tx, ty, h1_z;
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

      const Track* const track_pointer = track_flag ? tracklets : tracks;
      
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
      h0.z = hit_Zs[h0_num];

      ASSERT(h1_num < number_of_hits)
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
    for (int k=0; k<(sensor_data[SENSOR_DATA_HITNUMS + 2] + blockDim_sh_hit - 1) / blockDim_sh_hit; ++k) {
      
#if USE_SHARED_FOR_HITS
      __syncthreads();
      const int tid = threadIdx.y * blockDim.x + threadIdx.x;
      const int sh_hit_no = blockDim_sh_hit * k + tid;
      if (threadIdx.y < SH_HIT_MULT && sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2]) {
        const int h2_index = sensor_data[2] + sh_hit_no;

        // Coalesced memory accesses
        ASSERT(tid < blockDim_sh_hit)
        sh_hit_x[tid] = hit_Xs[h2_index];
        sh_hit_y[tid] = hit_Ys[h2_index];
        sh_hit_z[tid] = hit_Zs[h2_index];
      }
      __syncthreads();
#endif

      if (ttf_condition) {
        const int last_hit_h2 = min(blockDim_sh_hit * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
        for (int kk=blockDim_sh_hit * k; kk<last_hit_h2; ++kk) {
          
          const int h2_index = sensor_data[2] + kk;
#if USE_SHARED_FOR_HITS
          const int sh_h2_index = kk % blockDim_sh_hit;
          const Hit h2 {sh_hit_x[sh_h2_index], sh_hit_y[sh_h2_index], sh_hit_z[sh_h2_index]};
#else
          const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index], hit_Zs[h2_index]};
#endif

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
