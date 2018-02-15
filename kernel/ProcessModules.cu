#include "SearchByTriplet.cuh"

/**
 * @brief Processes modules in decreasing order with some stride
 */
__device__ void processModules(
  Module* module_data,
  const int starting_module,
  const int stride,
  bool* hit_used,
  const int* h0_candidates,
  const int* h2_candidates,
  const int number_of_modules,
  const int* module_hitStarts,
  const int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* module_Zs,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* sh_hit_lastPointer,
  unsigned int* tracks_insertPointer,
  int* tracks_to_follow,
  int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  const int number_of_hits,
  unsigned int* h1_rel_indices,
  unsigned int* local_number_of_hits
) {
  int first_module = starting_module;

  // Prepare s1 and s2 for the first iteration
  unsigned int prev_ttf;
  unsigned int last_ttf = 0;

  while (first_module >= 4) {

    __syncthreads();
    
    // Iterate in modules
    // Load in shared
    if (threadIdx.x < 3) {
      const int module_number = first_module - threadIdx.x * 2;
      module_data[threadIdx.x].hitStart = module_hitStarts[module_number];
      module_data[threadIdx.x].hitNums = module_hitNums[module_number];
      module_data[threadIdx.x].z = module_Zs[module_number];
    }
    else if (threadIdx.x == 4) {
      sh_hit_lastPointer[0] = 0;
    }

    prev_ttf = last_ttf;
    last_ttf = ttf_insertPointer[0];
    const unsigned int diff_ttf = last_ttf - prev_ttf;

    // Reset atomics
    local_number_of_hits[0] = 0;

    __syncthreads();

    // Track Forwarding
    trackForwarding(
      hit_Xs,
      hit_Ys,
      hit_used,
      tracks_insertPointer,
      ttf_insertPointer,
      weaktracks_insertPointer,
      module_data,
      diff_ttf,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      number_of_hits,
      first_module,
      module_Zs,
      module_hitStarts,
      module_hitNums
    );

    __syncthreads();

    // Seeding
    trackSeeding(
      hit_Xs,
      hit_Ys,
      module_data,
      h0_candidates,
      h2_candidates,
      hit_used,
      tracklets_insertPointer,
      ttf_insertPointer,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      local_number_of_hits
    );

    first_module -= stride;
  }

  __syncthreads();

  prev_ttf = last_ttf;
  last_ttf = ttf_insertPointer[0];
  const unsigned int diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (int i=0; i<(diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const unsigned int ttf_element = blockDim.x * i + threadIdx.y * blockDim.x + threadIdx.x;

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
