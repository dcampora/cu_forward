#include "../include/SearchByTriplet.cuh"

/**
 * @brief Fits hits to tracks.
 * 
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns FLT_MAX.
 */
__device__ float fitHitToTrack(
  const Hit& h0,
  const Hit& h2,
  const float predx,
  const float predy,
  const float scatterDenom2
) {
  // tolerances
  const float x_prediction = h0.x + predx;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < TOLERANCE;

  const float y_prediction = h0.y + predy;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < TOLERANCE;

  // Scatter
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatter = scatterNum * scatterDenom2;

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
  const float* hit_Zs,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const Module* module_data,
  const unsigned int diff_ttf,
  unsigned int* tracks_to_follow,
  unsigned int* weak_tracks,
  const unsigned int prev_ttf,
  Track* tracklets,
  Track* tracks,
  const unsigned int number_of_hits,
  const unsigned int first_module,
  const float* module_Zs,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums
) {
  // Assign a track to follow to each thread
  for (int i=0; i<(diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const unsigned int ttf_element = blockDim.x * i + threadIdx.x;
    if (ttf_element < diff_ttf) {
      const auto fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % TTF_MODULO];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const auto skipped_modules = (fulltrackno & 0x70000000) >> 28;
      auto trackno = fulltrackno & 0x0FFFFFFF;

      const Track* track_pointer = track_flag ? tracklets : tracks;
      
      ASSERT(track_pointer==tracklets ? trackno < number_of_hits : true)
      ASSERT(track_pointer==tracks ? trackno < MAX_TRACKS : true)
      auto t = track_pointer[trackno];

      // Load last two hits in h0, h1
      ASSERT(t.hitsNum < MAX_TRACK_SIZE)
      const auto h0_num = t.hits[t.hitsNum - 2];
      const auto h1_num = t.hits[t.hitsNum - 1];

      ASSERT(h0_num < number_of_hits)
      const Hit h0 {hit_Xs[h0_num], hit_Ys[h0_num]};
      const auto h0_z = hit_Zs[h0_num];

      ASSERT(h1_num < number_of_hits)
      const Hit h1 {hit_Xs[h1_num], hit_Ys[h1_num]};
      const auto h1_z = hit_Zs[h1_num];

      // Track forwarding over t, for all hits in the next module
      // Line calculations
      const auto td = 1.0f / (h1_z - h0_z);
      const auto txn = (h1.x - h0.x);
      const auto tyn = (h1.y - h0.y);
      const auto tx = txn * td;
      const auto ty = tyn * td;

      // Find the best candidate
      float best_fit = FLT_MAX;
      unsigned short best_h2;

      // Some constants of fitting
      const auto h2_z = module_data[2].z;
      const auto dz = h2_z - h0_z;
      const auto predx = tx * dz;
      const auto predy = ty * dz;
      const auto scatterDenom2 = 1.f / ((h2_z - h1_z) * (h2_z - h1_z));

      for (auto j=0; j<module_data[2].hitNums; ++j) {
        const auto h2_index = module_data[2].hitStart + j;
        const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index]};
        const auto fit = fitHitToTrack(
          h0,
          h2,
          predx,
          predy,
          scatterDenom2
        );
        const auto fit_is_better = fit < best_fit;
        best_fit = fit_is_better*fit + !fit_is_better*best_fit;
        best_h2 = fit_is_better*h2_index + !fit_is_better*best_h2;
      }

      // Condition for finding a h2
      if (best_fit != FLT_MAX) {
        // Mark h2 as used
        ASSERT(best_h2 < number_of_hits)
        hit_used[best_h2] = true;

        // Update the tracks to follow, we'll have to follow up
        // this track on the next iteration :)
        ASSERT(t.hitsNum < MAX_TRACK_SIZE)
        t.hits[t.hitsNum++] = best_h2;

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
        const auto ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // A track just skipped a module
      // We keep it for another round
      else if (skipped_modules <= MAX_SKIPPED_MODULES) {
        // Form the new mask
        trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

        // Add the tracks to the bag of tracks to_follow
        const auto ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // If there are only three hits in this track,
      // mark it as "doubtful"
      else if (t.hitsNum == 3) {
        const auto weakP = atomicAdd(weaktracks_insertPointer, 1);
        ASSERT(weakP < number_of_hits)
        weak_tracks[weakP] = trackno;
      }
      // In the "else" case, we couldn't follow up the track,
      // so we won't be track following it anymore.
    }
  }
}
