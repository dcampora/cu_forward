#include "SearchByTriplet.cuh"

__device__ void fillCandidates(
  int* hit_candidates,
  int* hit_h2_candidates,
  const int number_of_sensors,
  const int* sensor_hitStarts,
  const int* sensor_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const int* sensor_Zs
) {
  const int blockDim_product = blockDim.x * blockDim.y;
  int first_sensor = number_of_sensors - 1;
  while (first_sensor >= 2) {
    const int second_sensor = first_sensor - 2;

    const bool process_h1_candidates = first_sensor >= 4;
    const bool process_h2_candidates = first_sensor <= number_of_sensors - 3;

    // Sensor dependent calculations
    const int z_s0 = process_h2_candidates ? sensor_Zs[first_sensor + 2] : 0;
    const int z_s1 = sensor_Zs[first_sensor];
    const int z_s2 = process_h2_candidates ? sensor_Zs[second_sensor] : 0;

    // Iterate in all hits in z0
    for (int i=0; i<(sensor_hitNums[first_sensor] + blockDim_product - 1) / blockDim_product; ++i) {
      const int h0_element = blockDim_product * i + threadIdx.y * blockDim.x + threadIdx.x;
      bool inside_bounds = h0_element < sensor_hitNums[first_sensor];

      if (inside_bounds) {
        bool first_h1_found = false, last_h1_found = false;
        bool first_h2_found = false, last_h2_found = false;
        const int h0_index = sensor_hitStarts[first_sensor] + h0_element;
        int h1_index;
        const float h0_x = hit_Xs[h0_index];
        const int hitstarts_s2 = sensor_hitStarts[second_sensor];
        const int hitnums_s2 = sensor_hitNums[second_sensor];

        float xmin_h2, xmax_h2;
        if (process_h2_candidates) {
          // Note: Here, we take h0 as if it were h1, the rest
          // of the notation is fine.
          
          // Min and max possible x0s
          const float h_dist = fabs(z_s1 - z_s0);
          const float dxmax = PARAM_MAXXSLOPE_CANDIDATES * h_dist;
          const float x0_min = h0_x - dxmax;
          const float x0_max = h0_x + dxmax;

          // Min and max possible h1s for that h0
          float z2_tz = (((float) z_s2 - z_s0)) / (z_s1 - z_s0);
          float x = x0_max + (h0_x - x0_max) * z2_tz;
          xmin_h2 = x - PARAM_TOLERANCE_CANDIDATES;

          x = x0_min + (h0_x - x0_min) * z2_tz;
          xmax_h2 = x + PARAM_TOLERANCE_CANDIDATES;
        }
        
        if (first_sensor >= 4) {
          // Iterate in all hits in z1
          for (int h1_element=0; h1_element<hitnums_s2; ++h1_element) {
            inside_bounds = h1_element < hitnums_s2;

            if (inside_bounds) {
              h1_index = hitstarts_s2 + h1_element;
              const float h1_x = hit_Xs[h1_index];

              if (process_h1_candidates && !last_h1_found) {
                // Check if h0 and h1 are compatible
                const float h_dist = fabs(z_s2 - z_s1);
                const float dxmax = PARAM_MAXXSLOPE_CANDIDATES * h_dist;
                const bool tol_condition = fabs(h1_x - h0_x) < dxmax;
                
                // Find the first one
                if (!first_h1_found && tol_condition) {
                  ASSERT(2 * h0_index < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                  hit_candidates[2 * h0_index] = h1_index;
                  first_h1_found = true;
                }
                // The last one, only if the first one has already been found
                else if (first_h1_found && !tol_condition) {
                  ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                  hit_candidates[2 * h0_index + 1] = h1_index;
                  last_h1_found = true;
                }
              }

              if (process_h2_candidates && !last_h2_found) {
                if (!first_h2_found && h1_x > xmin_h2) {
                  ASSERT(2 * h0_index < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                  hit_h2_candidates[2 * h0_index] = h1_index;
                  first_h2_found = true;
                }
                else if (first_h2_found && h1_x > xmax_h2) {
                  ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                  hit_h2_candidates[2 * h0_index + 1] = h1_index;
                  last_h2_found = true;
                }
              }

              if ((!process_h1_candidates || last_h1_found) &&
                  (!process_h2_candidates || last_h2_found)) {
                break;
              }
            }
          }

          // Note: If first is not found, then both should be -1
          // and there wouldn't be any iteration
          if (process_h1_candidates && first_h1_found && !last_h1_found) {
            ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

            hit_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
          }

          if (process_h2_candidates && first_h2_found && !last_h2_found) {
            ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

            hit_h2_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
          }
        }
      }
    }

    --first_sensor;
  }
}
