#include "SearchByTriplet.cuh"

/**
 * @brief  First iteration of seeding
 * 
 * @detail Due to the higher amount of triplets and the
 *         abscence of flagged hits at this stage, there is
 *         a specialized algorithm for it
 */
__device__ void trackSeedingFirst(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  unsigned int* tracks_to_follow
) {
  // Some constants of the calculation below
  const auto dmax = MAX_SLOPE * (module_data[0].z - module_data[1].z);
  const auto scatterDenom2 = 1.f / ((module_data[2].z - module_data[1].z) * (module_data[2].z - module_data[1].z));
  const auto z2_tz = (module_data[2].z - module_data[0].z) / (module_data[1].z - module_data[0].z);

  // Adaptive number of xthreads and ythreads,
  // depending on number of hits in h1 to process

  // Process at a time a maximum of MAX_CONCURRENT_H1_FIRST_ITERATION elements
  const auto number_of_hits_h1 = module_data[1].hitNums;
  const auto last_iteration = (number_of_hits_h1 + MAX_CONCURRENT_H1_FIRST_ITERATION - 1) / MAX_CONCURRENT_H1_FIRST_ITERATION;
  const auto num_hits_last_iteration = ((number_of_hits_h1 - 1) % MAX_CONCURRENT_H1_FIRST_ITERATION) + 1;
  for (int i=0; i<last_iteration; ++i) {
    // The output we are searching for
    unsigned short best_h0 = 0;
    unsigned short best_h2 = 0;
    unsigned short h1_index = 0;
    float best_fit = FLT_MAX;

    // Assign an adaptive x and y id for the current thread depending on the load.
    // This is not trivial because:
    // 
    // - We want a MAX_CONCURRENT_H1_FIRST_ITERATION:
    //     It turns out the load is more evenly balanced if we distribute systematically
    //     the processing of one h1 across several threads (ie. h0_candidates x h2_candidates)
    //     
    // - The last iteration is quite different from the others:
    //     We have a variable number of hits in the last iteration
    //     
    const auto is_last_iteration = i==last_iteration-1;
    const auto block_dim_x = is_last_iteration*num_hits_last_iteration + 
                             !is_last_iteration*MAX_CONCURRENT_H1_FIRST_ITERATION;

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
    // Ie. if processing 30 h1 hits, with MAX_CONCURRENT_H1_FIRST_ITERATION = 8, #h1 in each iteration to process are:
    // {8, 8, 8, 6}
    // On the fourth iteration, we should start from 3*MAX_CONCURRENT_H1_FIRST_ITERATION (24 + thread_id_x)
    const auto h1_rel_index = i*MAX_CONCURRENT_H1_FIRST_ITERATION + thread_id_x;
    if (h1_rel_index < number_of_hits_h1) {
      // Fetch h1
      h1_index = module_data[1].hitStart + h1_rel_index;
      const Hit h1 {hit_Xs[h1_index], hit_Ys[h1_index]};

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
          // Fetch h0
          const Hit h0 {hit_Xs[h0_index], hit_Ys[h0_index]};

          // Finally, iterate over all h2 indices
          for (auto h2_index=h2_first_candidate; h2_index<h2_last_candidate; ++h2_index) {
            // Our triplet is h0_index, h1_index, h2_index
            // Fit it and check if it's better than what this thread had
            // for any triplet with h1
            const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index]};

            // Calculate prediction
            const auto x = h0.x + (h1.x - h0.x) * z2_tz;
            const auto y = h0.y + (h1.y - h0.y) * z2_tz;
            const auto dx = x - h2.x;
            const auto dy = y - h2.y;

            // Calculate fit
            const auto scatterNum = (dx * dx) + (dy * dy);
            const auto scatter = scatterNum * scatterDenom2;
            const auto condition = fabs(h1.x - h0.x) < dmax &&
                                   fabs(h1.y - h0.y) < dmax &&
                                   fabs(dx) < TOLERANCE &&
                                   fabs(dy) < TOLERANCE &&
                                   scatter < MAX_SCATTER_SEEDING &&
                                   scatter < best_fit;

            // Populate fit, h0 and h2 in case we have found a better one
            best_fit = condition*scatter + !condition*best_fit;
            best_h0 = condition*h0_index + !condition*best_h0;
            best_h2 = condition*h2_index + !condition*best_h2;
          }
        }
      }
    }

    shared_best_fits[threadIdx.x] = best_fit;

    // Due to shared_best_fits
    __syncthreads();

    // We have calculated block_dim_x hits
    // Find out if we (the current threadIdx.x) is the best,
    // and if so, create and add a track
    int winner_thread = -1;
    best_fit = FLT_MAX;
    for (int id_y=0; id_y<block_dim_y; ++id_y) {
      const int shared_address = id_y * block_dim_x + thread_id_x;
      const auto better_fit = shared_best_fits[shared_address] < best_fit;
      winner_thread = better_fit*shared_address + !better_fit*winner_thread;
      best_fit = better_fit*shared_best_fits[shared_address] + !better_fit*best_fit;
    }

    // If this condition holds, then necessarily best_fit < FLT_MAX
    if (threadIdx.x == winner_thread) {
      // Add the track to the bag of tracks
      const auto trackP = atomicAdd(tracklets_insertPointer, 1);
      // ASSERT(trackP < number_of_hits)
      tracklets[trackP] = Track {3, best_h0, h1_index, best_h2};

      // Add the tracks to the bag of tracks to_follow
      // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
      // and hence it is stored in tracklets
      const auto ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
      tracks_to_follow[ttfP] = 0x80000000 | trackP;
    }

    // Due to RAW between iterations shared_best_fits
    __syncthreads();
  }
}
