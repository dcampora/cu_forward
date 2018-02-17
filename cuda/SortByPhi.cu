#include "../include/SearchByTriplet.cuh"
#include "math_constants.h"

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__device__ void applyPermutation(
  unsigned short* permutation,
  const unsigned int number_of_hits,
  T* prev_container,
  T* new_container
) {
  // Apply permutation across all hits in the module (coalesced)
  for (unsigned int i=0; i<(number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto permutation_index = i*blockDim.x + threadIdx.x;
    if (permutation_index < number_of_hits) {
      const auto hit_index = permutation[permutation_index];
      new_container[permutation_index] = prev_container[hit_index];
    }
  }
}

/**
 * @brief Calculates phi for each hit
 */
__device__ void sortByPhi(
  const unsigned int number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  unsigned int* hit_IDs,
  int32_t* hit_temp,
  unsigned short* hit_permutation
) {
  // Let's work with new pointers
  // Note: It is important we populate later on in strictly
  //       the same order, to not lose data
  float* new_hit_Xs = (float*) hit_temp;
  float* new_hit_Ys = hit_Xs;
  float* new_hit_Zs = hit_Ys;
  unsigned int* new_hit_IDs = (unsigned int*) hit_Zs;

  // Apply permutation across all arrays
  applyPermutation(hit_permutation, number_of_hits, hit_Xs, new_hit_Xs);
  __syncthreads();
  applyPermutation(hit_permutation, number_of_hits, hit_Ys, new_hit_Ys);
  __syncthreads();
  applyPermutation(hit_permutation, number_of_hits, hit_Zs, new_hit_Zs);
  __syncthreads();
  applyPermutation(hit_permutation, number_of_hits, hit_IDs, new_hit_IDs);
}
