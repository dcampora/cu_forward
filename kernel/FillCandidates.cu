#include "SearchByTriplet.cuh"

__device__ void fillCandidates(
  int* h0_candidates,
  int* h2_candidates,
  const int number_of_sensors,
  const int* sensor_hitStarts,
  const int* sensor_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* sensor_Zs
) {
  // Notation is s0, s1, s2 in reverse order for each sensor
  // A hit in those is h0, h1, h2 respectively

  // Assign a module, h1 combination to each threadIdx.x
  int module_h1_counter = 0;
  for (int sensor_index=2; sensor_index<=49; ++sensor_index) {
    const auto s1_hitNums = sensor_hitNums[sensor_index];
    for (int i=0; i<(s1_hitNums + blockDim.x - 1) / blockDim.x; ++i) {
      const auto h1_rel_index = i*blockDim.x + threadIdx.x;

      if (h1_rel_index < s1_hitNums) {
        // Find for module sensor_index, hit h1_rel_index the candidates
        const auto s0_z = sensor_Zs[sensor_index+2];
        const auto s1_z = sensor_Zs[sensor_index];
        const auto s2_z = sensor_Zs[sensor_index-2];
        const auto s0_hitStarts = sensor_hitStarts[sensor_index+2];
        const auto s2_hitStarts = sensor_hitStarts[sensor_index-2];
        const auto s0_hitNums = sensor_hitNums[sensor_index+2];
        const auto s2_hitNums = sensor_hitNums[sensor_index-2];
        const auto h1_index = sensor_hitStarts[sensor_index] + h1_rel_index;
        const auto h1_x = hit_Xs[h1_index];

        // Calculate x limits in h0 and h2
        // Note: f0(z) = alpha*z
        //       f2(z) = (alpha+beta)*z
        const auto tolerance_s0 = PARAM_TOLERANCE_ALPHA * (s0_z - s1_z);
        const auto tolerance_s2 = (PARAM_TOLERANCE_ALPHA + PARAM_TOLERANCE_BETA) * (s1_z - s2_z);

        // Find candidates
        bool first_h0_found = false, last_h0_found = false;
        bool first_h2_found = false, last_h2_found = false;
        
        // Add h0 candidates
        for (int h0_index=s0_hitStarts; h0_index < s0_hitStarts + s0_hitNums; ++h0_index) {
          const auto h0_x = hit_Xs[h0_index];
          const bool tolerance_condition = fabs(h1_x - h0_x) < tolerance_s0;

          if (!first_h0_found && tolerance_condition) {
            h0_candidates[2*h1_index] = h0_index;
            first_h0_found = true;
          }
          else if (first_h0_found && !last_h0_found && !tolerance_condition) {
            h0_candidates[2*h1_index + 1] = h0_index;
            last_h0_found = true;
          }
        }
        if (first_h0_found && !last_h0_found) {
          h0_candidates[2*h1_index + 1] = s0_hitStarts + s0_hitNums;
        }

        // Add h2 candidates
        for (int h2_index=s2_hitStarts; h2_index < s2_hitStarts + s2_hitNums; ++h2_index) {
          const auto h2_x = hit_Xs[h2_index];
          const bool tolerance_condition = fabs(h1_x - h2_x) < tolerance_s2;

          if (!first_h2_found && tolerance_condition) {
            h2_candidates[2*h1_index] = h2_index;
            first_h2_found = true;
          }
          else if (first_h2_found && !last_h2_found && !tolerance_condition) {
            h2_candidates[2*h1_index + 1] = h2_index;
            last_h2_found = true;
          }
        }
        if (first_h2_found && !last_h2_found) {
          h2_candidates[2*h1_index + 1] = s2_hitStarts + s2_hitNums;
        }
      }
    }
  }
}
