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
  unsigned int* best_h0s,
  unsigned int* best_h2s,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  int* tracks_to_follow,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
) {
  // Initialize best_fits to MAX_FLOAT (coalesced write pattern)
  // We don't need to initialize best_h0s and best_h2s, since
  // they will only be looked up if best_fit != MAX_FLOAT
  for (int i=0; i<sensor_data[1].hitNums; ++i) {
    best_fits[threadIdx.x*MAX_NUMHITS_IN_MODULE + i] = MAX_FLOAT;
  }

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

  // Adaptive number of xthreads and ythreads,
  // depending on number of hits in h1 to process

  // Process at a time a maximum of MAX_CONCURRENT_H1 elements
  const auto number_of_hits_h1 = local_number_of_hits[0];
  const auto last_iteration = (number_of_hits_h1 + MAX_CONCURRENT_H1 - 1) / MAX_CONCURRENT_H1;
  const auto num_hits_last_iteration = ((number_of_hits_h1 - 1) % MAX_CONCURRENT_H1) + 1;
  for (int i=0; i<last_iteration; ++i) {
    // Assign an adaptive x and y id for the current thread depending on the load.
    // This is not trivial because:
    // 
    // - We want a MAX_CONCURRENT_H1:
    //     It turns out the load is more evenly balanced if we distribute systematically
    //     the processing of one h1 across several threads (ie. h0_candidates x h2_candidates)
    //     
    // - The last iteration is quite different from the others:
    //     We have a variable number of hits in the last iteration
    //     
    const auto is_last_iteration = i==last_iteration-1;
    const auto block_dim_x = is_last_iteration*num_hits_last_iteration + 
                             !is_last_iteration*MAX_CONCURRENT_H1;

    // Adapted local thread ID of each thread
    const auto thread_id_x = threadIdx.x % block_dim_x;
    const auto thread_id_y = threadIdx.x / block_dim_x;

    // block dim y is tricky because its size varies when we are in the last iteration
    // ie. blockDim.x=64, block_dim_x=10
    // We have threads until ... #60 {0,6}, #61 {1,6}, #62 {2,6}, #63 {3,6}
    // Therefore, threads {4,X}, {5,X}, {6,X}, {7,X}, {8,X}, {9,X} all have block_dim_y=6
    // whereas threads {0,X}, {1,X}, {2,X}, {3,X} have block_dim_y=7
    // We will call the last one the "numerous" block ({0,X}, {1,X}, {2,X}, {3,X})
    const auto is_in_numerous_block = thread_id_x < ((blockDim.x-1) % block_dim_x) + 1;
    const auto block_dim_y = is_in_numerous_block*((blockDim.x + block_dim_x - 1) / block_dim_x) +
                            !is_in_numerous_block*((blockDim.x-1) / block_dim_x);

    // Work with thread_id_x, thread_id_y and block_dim_y from now on
    // Each h1 is associated with a thread_id_x
    // 
    // Ie. if processing 30 h1 hits, with MAX_CONCURRENT_H1 = 8, #h1 in each iteration to process are:
    // {8, 8, 8, 6}
    // On the fourth iteration, we should start from 3*MAX_CONCURRENT_H1 (24 + thread_id_x)
    const auto h1_rel_rel_index = i*MAX_CONCURRENT_H1 + thread_id_x;
    if (h1_rel_rel_index < number_of_hits_h1) {
      // h1 index
      const auto h1_rel_index = h1_rel_indices[h1_rel_rel_index];
      const auto h1_index = sensor_data[1].hitStart + h1_rel_index;

      // Iterate over all h0, h2 combinations
      // Ignore used hits
      const auto h0_first_candidate = h0_candidates[2*h1_index];
      const auto h0_last_candidate = h0_candidates[2*h1_index + 1];
      const auto h2_first_candidate = h2_candidates[2*h1_index];
      const auto h2_last_candidate = h2_candidates[2*h1_index + 1];

      // Iterate over h0 with thread_id_y
      const auto h0_num_candidates = h0_last_candidate - h0_first_candidate;
      for (int j=0; j<(h0_num_candidates + block_dim_y - 1) / block_dim_y; ++j) {
        const auto h0_rel_candidate = j*block_dim_y + thread_id_y;
        if (h0_rel_candidate < h0_num_candidates) {
          const auto h0_index = h0_first_candidate + h0_rel_candidate;
          if (!hit_used[h0_index]) {
            // Finally, iterate over all h2 indices
            for (int h2_index=h2_first_candidate; h2_index<h2_last_candidate; ++h2_index) {
              if (!hit_used[h2_index]) {
                const auto best_fits_index = thread_id_y*MAX_NUMHITS_IN_MODULE + h1_rel_index;

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
                                       scatter < best_fits[best_fits_index];

                // Populate fit, h0 and h2 in case we have found a better one
                best_fits[best_fits_index] = condition*scatter + !condition*best_fits[best_fits_index];
                best_h0s[best_fits_index] = condition*h0_index + !condition*best_h0s[best_fits_index];
                best_h2s[best_fits_index] = condition*h2_index + !condition*best_h2s[best_fits_index];
              }
            }
          }
        }
      }
    }
  }
  
  // Due to best_fits
  __syncthreads();

  // We have calculated all triplets
  // Assign a thread for each h1
  // const auto last_iteration = (number_of_hits_h1 + blockDim.x - 1) / blockDim.x;
  // const auto num_hits_last_iteration = number_of_hits_h1 % blockDim.x;
  for (int i=0; i<last_iteration; ++i) {
    const auto is_last_iteration = i == last_iteration - 1;
    const auto num_hits_iteration = is_last_iteration*num_hits_last_iteration + 
                                    !is_last_iteration*MAX_CONCURRENT_H1;

    const auto h1_rel_rel_index = i*blockDim.x + threadIdx.x;
    if (h1_rel_rel_index < number_of_hits_h1) {
      const auto h1_rel_index = h1_rel_indices[h1_rel_rel_index];
      const auto h1_index = sensor_data[1].hitStart + h1_rel_index;

      // Traverse all fits done by all threads,
      // and find the best one
      float best_fit = MAX_FLOAT;
      unsigned int best_h0 = 0;
      unsigned int best_h2 = 0;

      // Several threads have calculated the fits
      // Use a simplified block_dim_y: The highest, regardless of the last block dimensions
      const auto block_dim_y_simplified = (blockDim.x + num_hits_last_iteration - 1) / num_hits_last_iteration;
      for (int i=0; i<block_dim_y_simplified; ++i) {
        const auto current_index = i*MAX_NUMHITS_IN_MODULE + h1_rel_index;
        if (best_fits[current_index] < best_fit) {
          best_fit = best_fits[current_index];
          best_h0 = best_h0s[current_index];
          best_h2 = best_h2s[current_index];
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