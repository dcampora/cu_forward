#include "../include/SearchByTriplet.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void searchByTriplet(
  Track* dev_tracks,
  const char* dev_input,
  unsigned int* dev_tracks_to_follow,
  bool* dev_hit_used,
  int* dev_atomicsStorage,
  Track* dev_tracklets,
  unsigned int* dev_weak_tracks,
  unsigned int* dev_event_offsets,
  unsigned int* dev_hit_offsets,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices,
  float* dev_hit_phi,
  int32_t* dev_hit_temp
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const unsigned int event_number = blockIdx.x;
  const unsigned int events_under_process = gridDim.x;
  const unsigned int tracks_offset = event_number * MAX_TRACKS;

  // Pointers to data within the event
  const unsigned int data_offset = dev_event_offsets[event_number];
  const unsigned int* no_modules = (const unsigned int*) &dev_input[data_offset];
  const unsigned int* no_hits = (const unsigned int*) (no_modules + 1);
  const float* module_Zs = (const float*) (no_hits + 1);
  const unsigned int number_of_modules = no_modules[0];
  const unsigned int number_of_hits = no_hits[0];
  const unsigned int* module_hitStarts = (const unsigned int*) (module_Zs + number_of_modules);
  const unsigned int* module_hitNums = (const unsigned int*) (module_hitStarts + number_of_modules);
  int32_t* hit_temp = (int32_t*) (module_hitNums + number_of_modules);
  float* hit_Ys = (float*) (hit_temp + number_of_hits);
  float* hit_Zs = (float*) (hit_Ys + number_of_hits);
  unsigned int* hit_IDs = (unsigned int*) (hit_Zs + number_of_hits);

  // Per event datatypes
  Track* tracks = dev_tracks + tracks_offset;
  unsigned int* tracks_insertPointer = (unsigned int*) dev_atomicsStorage + event_number;

  // Per side datatypes
  const unsigned int hit_offset = dev_hit_offsets[event_number];
  bool* hit_used = dev_hit_used + hit_offset;
  short* h0_candidates = dev_h0_candidates + hit_offset * 2;
  short* h2_candidates = dev_h2_candidates + hit_offset * 2;
  float* hit_Phis = (float*) (dev_hit_phi + hit_offset);
  float* hit_Xs = (float*) (dev_hit_temp + hit_offset);

  unsigned int* tracks_to_follow = dev_tracks_to_follow + event_number * TTF_MODULO;
  unsigned int* weak_tracks = dev_weak_tracks + hit_offset;
  Track* tracklets = dev_tracklets + hit_offset;
  unsigned short* h1_rel_indices = dev_rel_indices + event_number * MAX_NUMHITS_IN_MODULE;

  // Initialize variables according to event number and module side
  // Insert pointers (atomics)
  const int ip_shift = events_under_process + event_number * NUM_ATOMICS;
  unsigned int* weaktracks_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 1;
  unsigned int* tracklets_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 2;
  unsigned int* ttf_insertPointer = (unsigned int*) dev_atomicsStorage + ip_shift + 3;
  unsigned int* local_number_of_hits = (unsigned int*) dev_atomicsStorage + ip_shift + 4;

  // Shared memory
  __shared__ float shared_best_fits [NUMTHREADS_X];
  __shared__ int module_data [6];

#if DO_REPEATED_EXECUTION
  for (int repetitions=0; repetitions<REPEAT_ITERATIONS; ++repetitions) {
  // Initialize hit_used
  for (int i=0; i<(number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto index = i*blockDim.x + threadIdx.x;
    if (index < number_of_hits) {
      hit_used[index] = false;
    }
  }
  // Initialize atomics
  tracks_insertPointer[0] = 0;
  if (threadIdx.x < NUM_ATOMICS-1) {
    dev_atomicsStorage[threadIdx.x + ip_shift + 1] = 0;
  }
#endif

  // Fill candidates for both sides
  fillCandidates(
    h0_candidates,
    h2_candidates,
    module_hitStarts,
    module_hitNums,
    hit_Phis
  );

  // Process modules
  processModules(
    (Module*) &module_data[0],
    (float*) &shared_best_fits[0],
    number_of_modules-1,
    1,
    hit_used,
    h0_candidates,
    h2_candidates,
    number_of_modules,
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    module_Zs,
    weaktracks_insertPointer,
    tracklets_insertPointer,
    ttf_insertPointer,
    tracks_insertPointer,
    tracks_to_follow,
    weak_tracks,
    tracklets,
    tracks,
    number_of_hits,
    h1_rel_indices,
    local_number_of_hits
  );

  __syncthreads();

  // Process left weak tracks
  weakTracksAdder(
    (int*) &shared_best_fits[0],
    weaktracks_insertPointer,
    tracks_insertPointer,
    weak_tracks,
    tracklets,
    tracks,
    hit_used
  );

#if DO_REPEATED_EXECUTION
  __syncthreads();
  }
#endif
}
