#include "SearchByTriplet.cuh"

/**
 * @brief Creates the tracks seeds from unused triplets.
 * 
 * @detail Uses "even share" across all threads to do the computation
 *         and joins all results afterwards
 */
__device__ void trackSeeding(
#if USE_SHARED_FOR_HITS
  float* sh_hit_x,
  float* sh_hit_y,
#endif
  const float* hit_Xs,
  const float* hit_Ys,
  const Sensor* sensor_data,
  int* h0_candidates,
  unsigned int* max_numhits_to_process,
  bool* hit_used,
  int* h2_candidates,
  const int blockDim_sh_hit,
  float* best_fits,
  unsigned int* best_h0s,
  unsigned int* best_h2s,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  int* tracks_to_follow,
  unsigned int* local_number_of_hits,
  unsigned int* local_unused_hits,
  unsigned int number_of_hits
) {
  // Initialize best_fits to MAX_FLOAT (coalesced write pattern)
  // We don't need to initialize best_h0s and best_h2s, since
  // they will only be looked up if best_fit != MAX_FLOAT
  for (int i=0; i<MAX_NUMHITS_IN_MODULE; ++i) {
    best_fits[i*blockDim.x + threadIdx.x] = MAX_FLOAT;
  }

  // Populate unused hit arrays
  for (int module_number=0; module_number<3; ++module_number) {
    for (int i=0; i<(sensor_data[module_number].hitNums + blockDim.x - 1) / blockDim.x; ++i) {
      const auto rel_index = i * blockDim.x + threadIdx.x;
      if (rel_index < sensor_data[module_number].hitNums) {
        const auto index = sensor_data[module_number].hitStart + rel_index;
        if (!hit_used[index]) {
          // Add to local sensor hit array
          const auto current_hit = atomicAdd(local_number_of_hits + module_number, 1);
          local_unused_hits[module_number*MAX_NUMHITS_IN_MODULE + current_hit] = index;
        }
      }
    }
  }

  // Due to atomics
  __syncthreads();

  // Order each array by hit_Xs
  // Let us use merge sort
  for (int module_number=0; module_number<3; ++module_number) {
    const unsigned int length = local_number_of_hits[module_number];
    unsigned int* array = local_unused_hits + module_number*MAX_NUMHITS_IN_MODULE;
    unsigned int* temp_array = local_unused_hits + 3*MAX_NUMHITS_IN_MODULE;

    int n = 1;
    while (n <= length) {
      const auto starting_hit = threadIdx.x*n*2;
      
      // Compare n numbers from starting hit
      // with n numbers from starting hit + n
      auto index_0 = starting_hit;
      auto index_1 = starting_hit + n;
      auto index_temp = starting_hit;
      const auto index_0_subset_end = starting_hit + n;
      const auto subset_end = starting_hit + 2*n;

      if (index_1 < length) {
        while (index_temp < subset_end && index_temp < length) {
          if (index_0 < index_0_subset_end && index_1 < subset_end) {
            const auto local_index_0 = array[index_0];
            const auto local_index_1 = array[index_1];
            ASSERT(local_index_0 < number_of_hits && local_index_1 < number_of_hits);
            const auto condition = hit_Xs[local_index_0] <= hit_Xs[local_index_1];
            // Conditional assignments
            temp_array[index_temp] = condition*array[index_0] + !condition*array[index_1];
            index_0 = condition*(index_0+1) + !condition*index_0;
            index_1 = condition*index_1 + !condition*(index_1+1);
            ++index_temp;
          }
          else if (index_0 < index_0_subset_end) {
            temp_array[index_temp] = array[index_0];
            ++index_0;
            ++index_temp;
          }
          else { // index_1 < subset_end
            temp_array[index_temp] = array[index_1];
            ++index_1;
            ++index_temp;
          }
        }
      }
      
      // Copy temp array to array
      __syncthreads();

      for (int i=0; i<(length + blockDim.x - 1) / blockDim.x; ++i) {
        const auto index = i * blockDim.x + threadIdx.x;
        if (index < length) {
          array[index] = temp_array[index];
        }
      }

      // Due to array reordering
      __syncthreads();

      n *= 2;
    }
  }

  // A syncthreads already ocurred, so we can
  // safely assume the arrays are ordered by x

  // For ease of use, define some pointers
  const auto s0_local_unused_hits = local_unused_hits;
  const auto s1_local_unused_hits = local_unused_hits + MAX_NUMHITS_IN_MODULE;
  const auto s2_local_unused_hits = local_unused_hits + 2*MAX_NUMHITS_IN_MODULE;

  // Fill candidates
  // Assign a thread for each h1
  for (int i=0; i<(local_number_of_hits[1] + blockDim.x - 1) / blockDim.x; ++i) {
    const auto h1_rel_index = i*blockDim.x + threadIdx.x;
    if (h1_rel_index < local_number_of_hits[1]) {
      const auto h1_index = s1_local_unused_hits[h1_rel_index];
      const auto h1_x = hit_Xs[h1_index];

      // Calculate x limits in h0 and h2
      // Note: f0(z) = alpha*z
      //       f2(z) = (alpha+beta)*z
      const auto tolerance_s0 = PARAM_TOLERANCE_ALPHA * (sensor_data[0].z - sensor_data[1].z);
      const auto tolerance_s2 = (PARAM_TOLERANCE_ALPHA + PARAM_TOLERANCE_BETA) * (sensor_data[1].z - sensor_data[2].z);

      // Find candidates
      bool first_h0_found = false, last_h0_found = false;
      bool first_h2_found = false, last_h2_found = false;

      // Add h0 candidates
      for (int h0_rel_index=0; h0_rel_index < local_number_of_hits[0]; ++h0_rel_index) {
        const auto h0_index = s0_local_unused_hits[h0_rel_index];
        const auto h0_x = hit_Xs[h0_index];
        const bool tolerance_condition = fabs(h1_x - h0_x) < tolerance_s0;

        if (!first_h0_found && tolerance_condition) {
          h0_candidates[2*h1_rel_index] = h0_rel_index;
          first_h0_found = true;
        }
        else if (first_h0_found && !last_h0_found && !tolerance_condition) {
          h0_candidates[2*h1_rel_index + 1] = h0_rel_index;
          last_h0_found = true;
        }
      }
      if (first_h0_found && !last_h0_found) {
        h0_candidates[2*h1_rel_index + 1] = local_number_of_hits[0];
      }
      else if (!first_h0_found && !last_h0_found) {
        h0_candidates[2*h1_rel_index] = -1;
        h0_candidates[2*h1_rel_index + 1] = -1;
      }

      // Add h2 candidates
      for (int h2_rel_index=0; h2_rel_index < local_number_of_hits[2]; ++h2_rel_index) {
        const auto h2_index = s2_local_unused_hits[h2_rel_index];
        const auto h2_x = hit_Xs[h2_index];
        const bool tolerance_condition = fabs(h1_x - h2_x) < tolerance_s2;

        if (!first_h2_found && tolerance_condition) {
          h2_candidates[2*h1_rel_index] = h2_rel_index;
          first_h2_found = true;
        }
        else if (first_h2_found && !last_h2_found && !tolerance_condition) {
          h2_candidates[2*h1_rel_index + 1] = h2_rel_index;
          last_h2_found = true;
        }
      }
      if (first_h2_found && !last_h2_found) {
        h2_candidates[2*h1_rel_index + 1] = local_number_of_hits[2];
      }
      else if (!first_h2_found && !last_h2_found) {
        h2_candidates[2*h1_rel_index] = -1;
        h2_candidates[2*h1_rel_index + 1] = -1;
      }
    }
  }

  // Due to candidates
  __syncthreads();

  // Populate an array with relative indices to s1_local_unused_hits that have candidates
  // Repurpose local_unused_hits + 3*MAX_NUMHITS_IN_MODULE
  const auto ordered_h1s_with_candidates = local_unused_hits + 3*MAX_NUMHITS_IN_MODULE;
  for (int i=0; i<(local_number_of_hits[1] + blockDim.x - 1) / blockDim.x; ++i) {
    const auto h1_rel_index = i*blockDim.x + threadIdx.x;
    if (h1_rel_index < local_number_of_hits[1]) {
      // This array should be something like:
      // {0, 1, 2, 4, 5, 7, ...}
      // ie. if there are missing relative indices, then that hit should be ignored
      // Use each thread to fill in the h1_rel_index position of this array
      int found_h1_with_candidates = -1;
      for (int j=0; j<local_number_of_hits[1]; ++j) {
        const auto h0_first_candidate = h0_candidates[2*j];
        const auto h2_first_candidate = h2_candidates[2*j];
        if (h0_first_candidate!=-1 && h2_first_candidate!=-1) {
          ++found_h1_with_candidates;
        }

        if (found_h1_with_candidates==h1_rel_index) {
          ordered_h1s_with_candidates[h1_rel_index] = j;
          atomicAdd(local_number_of_hits + 3, 1);
          break;
        }
      }
    }
  }

  // Due to ordered_h1s_with_candidates
  __syncthreads();

  // Candidates is now an ordered array of unused s0 and s2 hits
  // Count how many hits in total we need to process
  unsigned int total_number_of_triplets_to_process = 0;
  for (int h1_with_cand_index=0; h1_with_cand_index<local_number_of_hits[3]; ++h1_with_cand_index) {
    const auto h1_rel_index = ordered_h1s_with_candidates[h1_with_cand_index];
    const auto num_h0_candidates = h0_candidates[2*h1_rel_index+1] - h0_candidates[2*h1_rel_index];
    const auto num_h2_candidates = h2_candidates[2*h1_rel_index+1] - h2_candidates[2*h1_rel_index];
    total_number_of_triplets_to_process += num_h0_candidates * num_h2_candidates;
  }

  // Get the number of triplets we are going to process with this thread, and the starting triplet
  const unsigned int number_of_triplets_to_process = ((total_number_of_triplets_to_process - 1) / blockDim.x) + 1;
  const unsigned int starting_triplet_number = number_of_triplets_to_process * threadIdx.x;

  // Fetch the first h0, h1, h2 rel indices to work with
  unsigned int h0_rel_index = 0;
  unsigned int h1_with_cand_index = 0;
  unsigned int h2_rel_index = 0;
  unsigned int current_triplet = 0;
  for (int temp_h1_with_cand_index=0; temp_h1_with_cand_index<local_number_of_hits[3]; ++temp_h1_with_cand_index) {
    const auto temp_h1_rel_index = ordered_h1s_with_candidates[temp_h1_with_cand_index];
    const auto first_h0_candidate = h0_candidates[2*temp_h1_rel_index];
    const auto last_h0_candidate = h0_candidates[2*temp_h1_rel_index + 1];
    const auto first_h2_candidate = h2_candidates[2*temp_h1_rel_index];
    const auto last_h2_candidate = h2_candidates[2*temp_h1_rel_index + 1];

    for (int temp_h2_rel_index=first_h2_candidate; temp_h2_rel_index<last_h2_candidate; ++temp_h2_rel_index) {
      for (int temp_h0_rel_index=first_h0_candidate; temp_h0_rel_index<last_h0_candidate; ++temp_h0_rel_index) {
        if (starting_triplet_number == current_triplet++) {
          h0_rel_index = temp_h0_rel_index;
          h1_with_cand_index = temp_h1_with_cand_index;
          h2_rel_index = temp_h2_rel_index;
        }
      }
    }
  }

  // Test whether this syncthreads help or not
  // From now on, all processing is non-divergent
  __syncthreads();

  // Some constants of the calculation below
  const auto dymax = (PARAM_TOLERANCE_ALPHA + PARAM_TOLERANCE_BETA) * (sensor_data[0].z - sensor_data[1].z);
  const auto scatterDenom = 1.f / (sensor_data[2].z - sensor_data[1].z);
  const auto z2_tz = (sensor_data[2].z - sensor_data[0].z) / (sensor_data[1].z - sensor_data[0].z);

  // Iterate for as many triplets as requested
  current_triplet = 0;
  while (current_triplet < number_of_triplets_to_process &&
         starting_triplet_number + current_triplet < total_number_of_triplets_to_process) {
    // Our triplet is h0_rel_index, h1_rel_index, h2_rel_index
    // Fit it and check if it's better than what this thread had
    // for any triplet with h1
    const auto h1_rel_index = ordered_h1s_with_candidates[h1_with_cand_index];
    const auto h0_index = s0_local_unused_hits[h0_rel_index];
    const auto h1_index = s1_local_unused_hits[h1_rel_index];
    const auto h2_index = s2_local_unused_hits[h2_rel_index];
    const Hit h0 {hit_Xs[h0_index], hit_Ys[h0_index]};
    const Hit h1 {hit_Xs[h1_index], hit_Ys[h1_index]};
    const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index]};
    const auto best_fits_index = h1_rel_index*blockDim.x + threadIdx.x;

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
      
    // Check if it's better than what we had,
    // and in that case replace the best with scatter, h0_index, h2_index
    best_fits[best_fits_index] = condition*scatter + !condition*best_fits[best_fits_index];
    best_h0s[best_fits_index] = condition*h0_index + !condition*best_h0s[best_fits_index];
    best_h2s[best_fits_index] = condition*h2_index + !condition*best_h2s[best_fits_index];

    // Move on to the next triplet
    ++current_triplet;

    // Use conditional assignments
    const bool is_h0_last = h0_rel_index == h0_candidates[2*h1_rel_index + 1] - 1;
    const bool is_h2_last = h2_rel_index == h2_candidates[2*h1_rel_index + 1] - 1;

    // Note: h0_candidates and h2_candidates should not get out of bounds here
    //       It's probably okay to just allocate 2 more uints for each and leave this code as is
    const unsigned int current_starting_h0_candidate = h0_candidates[2*(h1_rel_index)];
    const auto next_h1 = ordered_h1s_with_candidates[h1_with_cand_index+1];
    const unsigned int next_starting_h0_candidate = h0_candidates[2*(next_h1)];
    const unsigned int next_starting_h2_candidate = h2_candidates[2*(next_h1)];

    // Note: In case we are at the last h0, h1, h2 triplet, the condition of the while loop will be false
    h0_rel_index = !is_h0_last*(h0_rel_index+1) +
                   is_h0_last*!is_h2_last*current_starting_h0_candidate +
                   is_h0_last*is_h2_last*next_starting_h0_candidate;

    h2_rel_index = !is_h0_last*h2_rel_index +
                   is_h0_last*!is_h2_last*(h2_rel_index + 1) +
                   is_h0_last*is_h2_last*next_starting_h2_candidate;

    h1_with_cand_index = !is_h0_last*h1_with_cand_index + 
                   is_h0_last*!is_h2_last*h1_with_cand_index + 
                   is_h0_last*is_h2_last*(h1_with_cand_index+1);
  }

  __syncthreads();

  // We have calculated all triplets
  // Assign a thread for each h1
  for (int i=0; i<(local_number_of_hits[3] + blockDim.x - 1) / blockDim.x; ++i) {
    const auto h1_with_cand_index = i*blockDim.x + threadIdx.x;
    if (h1_with_cand_index < local_number_of_hits[3]) {
      const auto h1_rel_index = ordered_h1s_with_candidates[h1_with_cand_index];
      const auto h1_index = s1_local_unused_hits[h1_rel_index];

      // Traverse all fits done by all threads,
      // and find the best one
      float best_fit = MAX_FLOAT;
      unsigned int best_h0 = 0;
      unsigned int best_h2 = 0;

      for (int i=0; i<blockDim.x; ++i) {
        const auto current_index = h1_rel_index*blockDim.x + i;
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
