#include "SearchByTriplet.cuh"

__device__ void trackSeeding(
  const float* hit_Xs,
  const float* hit_Ys,
  const Sensor* sensor_data,
  const int* h0_candidates,
  unsigned int* max_numhits_to_process,
  bool* hit_used,
  const int* h2_candidates,
  const int blockDim_sh_hit,
  float* best_fits,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  int* tracks_to_follow,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
) {
  // Add to an array all non-used h1 hits with candidates
  for (int i=0; i<(sensor_data[1].hitNums + blockDim.x - 1) / blockDim.x; ++i) {
    const auto h1_rel_index = i*blockDim.x + threadIdx.x;
    if (h1_rel_index < sensor_data[1].hitNums) {
      const auto h1_index = sensor_data[1].hitStart + h1_rel_index;
      const auto h0_first_candidate = h0_candidates[2*h1_index];
      const auto h2_first_candidate = h2_candidates[2*h1_index];
      if (!hit_used[h1_index] && h0_first_candidate!=-1 && h2_first_candidate!=-1) {
        const auto current_hit = atomicAdd(local_number_of_hits, 1);
        h1_rel_indices[current_hit] = h1_rel_index;
      }
    }
  }

  // Due to h1_rel_indices and best_fits
  __syncthreads();

  // Some constants of the calculation below
  const auto dymax = (PARAM_TOLERANCE_ALPHA + PARAM_TOLERANCE_BETA) * (sensor_data[0].z - sensor_data[1].z);
  const auto scatterDenom = 1.f / (sensor_data[2].z - sensor_data[1].z);
  const auto z2_tz = (sensor_data[2].z - sensor_data[0].z) / (sensor_data[1].z - sensor_data[0].z);

  // Simple implementation: Each h1 is associated with a thread
  // There may be threads that have no work to do
  for (int i=0; i<(local_number_of_hits[0] + blockDim.x - 1) / blockDim.x; ++i) {
    const auto h1_rel_rel_index = i*blockDim.x + threadIdx.x;
    if (h1_rel_rel_index < local_number_of_hits[0]) {
      // The solution we are searching
      float best_fit = MAX_FLOAT;
      unsigned int best_h0 = 0;
      unsigned int best_h2 = 0;

      const auto h1_rel_index = h1_rel_indices[h1_rel_rel_index];
      const auto h1_index = sensor_data[1].hitStart + h1_rel_index;

      // Iterate over all h0, h2 combinations
      // Ignore used hits
      const auto h0_first_candidate = h0_candidates[2*h1_index];
      const auto h0_last_candidate = h0_candidates[2*h1_index + 1];
      const auto h2_first_candidate = h2_candidates[2*h1_index];
      const auto h2_last_candidate = h2_candidates[2*h1_index + 1];
      for (int h0_rel_index=h0_first_candidate; h0_rel_index<h0_last_candidate; ++h0_rel_index) {
        const auto h0_index = sensor_data[0].hitStart + h0_rel_index;
        if (!hit_used[h0_index]) {
          for (int h2_rel_index=h2_first_candidate; h2_rel_index<h2_last_candidate; ++h2_rel_index) {
            const auto h2_index = sensor_data[2].hitStart + h2_rel_index;
            if (!hit_used[h2_index]) {
              // Our triplet is h0_index, h1_index, h2_index
              // Fit it and check if it's better than what this thread had
              // for any triplet with h1
              const Hit h0 {hit_Xs[h0_index], hit_Ys[h0_index]};
              const Hit h1 {hit_Xs[h1_index], hit_Ys[h1_index]};
              const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index]};

              // Calculate prediction
              const auto x = h0.x + (h1.x - h0.x) * z2_tz;
              const auto y = h0.y + (h1.y - h0.y) * z2_tz;
              const auto dx = x - h2.x;
              const auto dy = y - h2.y;

              // Calculate fit
              const auto scatterNum = (dx * dx) + (dy * dy);
              const auto scatter = scatterNum * scatterDenom * scatterDenom;
              const auto condition = fabs(h1.y - h0.y) < dymax &&
                                     fabs(dx) < PARAM_TOLERANCE &&
                                     fabs(dy) < PARAM_TOLERANCE &&
                                     scatter < MAX_SCATTER &&
                                     scatter < best_fit;


              best_fit = condition*scatter + !condition*best_fit;
              best_h0 = condition*h0_index + !condition*best_h0;
              best_h2 = condition*h2_index + !condition*best_h2;
            }
          }
        }
      }

      // In case we found a best fit < MAX_FLOAT,
      // add the triplet best_h0, h1_index, best_h2 to our forming tracks
      if (best_fit < MAX_FLOAT) {
        // Add the track to the bag of tracks
        const unsigned int trackP = atomicAdd(tracklets_insertPointer, 1);
        // ASSERT(trackP < number_of_hits)
        tracklets[trackP] = Track {3, best_h0, h1_index, best_h2};

        // Add the tracks to the bag of tracks to_follow
        // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
        // and hence it is stored in tracklets
        const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = 0x80000000 | trackP;
      }
    }
  }
}
  // // Initialize best_fits to MAX_FLOAT (coalesced write pattern)
  // // We don't need to initialize best_h0s and best_h2s, since
  // // they will only be looked up if best_fit != MAX_FLOAT
  // for (int i=0; i<MAX_NUMHITS_IN_MODULE; ++i) {
  //   best_fits[i*blockDim.x + threadIdx.x] = MAX_FLOAT;
  // }

  // Adaptive number of xthreads and ythreads,
  // depending on number of hits in h1 to process

  // // Process at a time a maximum of blockDim.x elements
  // const auto last_iteration = ((local_number_of_hits[0] - 1) / blockDim.x) + 1;
  // const auto num_hits_last_iteration = local_number_of_hits[0] % blockDim.x;
  // for (int i=0; i<last_iteration; ++i) {
  //   // Assign a x and y for the current thread
  //   const auto is_last_iteration = i==last_iteration-1;
  //   const auto thread_id_x = is_last_iteration*(threadIdx.x % num_hits_last_iteration) + !is_last_iteration*threadIdx.x;
  //   const auto thread_id_y = is_last_iteration*(threadIdx.x / num_hits_last_iteration) + !is_last_iteration*1;
  //   const auto block_dim_x = is_last_iteration*num_hits_last_iteration + !is_last_iteration*blockDim.x;
  //   const auto block_dim_y = is_last_iteration*num_hits_last_iteration + !is_last_iteration*1;

  //   // Work with thread_id_x and thread_id_y from now on

  // }