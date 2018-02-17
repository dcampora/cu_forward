#include "../include/CalculatePhiAndSort.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void calculatePhiAndSort(
  const char* dev_input,
  unsigned int* dev_event_offsets,
  unsigned int* dev_hit_offsets,
  float* dev_hit_phi,
  int32_t* dev_hit_temp,
  unsigned short* dev_hit_permutations
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const unsigned int event_number = blockIdx.x;

  // Pointers to data within the event
  const unsigned int data_offset = dev_event_offsets[event_number];
  const unsigned int* no_modules = (const unsigned int*) &dev_input[data_offset];
  const unsigned int* no_hits = (const unsigned int*) (no_modules + 1);
  const float* module_Zs = (const float*) (no_hits + 1);
  const unsigned int number_of_modules = no_modules[0];
  const unsigned int number_of_hits = no_hits[0];
  const unsigned int* module_hitStarts = (const unsigned int*) (module_Zs + number_of_modules);
  const unsigned int* module_hitNums = (const unsigned int*) (module_hitStarts + number_of_modules);
  unsigned int* hit_IDs = (unsigned int*) (module_hitNums + number_of_modules);
  float* hit_Xs = (float*) (hit_IDs + number_of_hits);
  float* hit_Ys = (float*) (hit_Xs + number_of_hits);
  float* hit_Zs = (float*) (hit_Ys + number_of_hits);

  // Per side datatypes
  const unsigned int hit_offset = dev_hit_offsets[event_number];
  float* hit_Phis = (float*) (dev_hit_phi + hit_offset);
  int32_t* hit_temp = (int32_t*) (dev_hit_temp + hit_offset);
  unsigned short* hit_permutations = dev_hit_permutations + hit_offset;

  // Calculate phi and populate hit_permutations
  calculatePhi(
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Phis,
    hit_permutations
  );

  // Due to phi RAW
  __syncthreads();

  // Sort by phi
  sortByPhi(
    number_of_hits,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    hit_IDs,
    hit_temp,
    hit_permutations
  );
}
