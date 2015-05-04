
#include "KernelFits.cuh"

/**
 * @brief Gives the fit between h0, h1 and h2
 * @details The result is given in a float. MAX_FLOAT is
 *          used as an upper limit. The lower it is, the better the fit is.
 *          
 * @param  h0 
 * @param  h1 
 * @param  h2 
 * @return    
 */
__device__ float fitHits(const Hit& h0, const Hit& h1, const Hit &h2) {
  // Max dx, dy permissible over next hit

  // First approximation -
  // With the sensor z, instead of the hit z
  const float z2_tz = (h2.z - h0.z) / (h1.z - h0.z);
  const float x = h0.x + (h1.x - h0.x) * z2_tz;
  const float y = h0.y + (h1.y - h0.y) * z2_tz;

  const float dx = x - h2.x;
  const float dy = y - h2.y;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2.z - h1.z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool condition = scatter < MAX_SCATTER;

  return condition * scatter + !condition * MAX_FLOAT;
}

/**
 * @brief Fits hits to tracks.
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns MAX_FLOAT.
 * 
 * @param tx 
 * @param ty 
 * @param h0 
 * @param h1_z
 * @param h2 
 * @return 
 */
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const float h1_z, const Hit& h2) {
  // tolerances
  const float dz = h2.z - h0.z;
  const float x_prediction = h0.x + tx * dz;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < PARAM_TOLERANCE;

  const float y_prediction = h0.y + ty * dz;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < PARAM_TOLERANCE;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2.z - h1_z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool scatter_condition = scatter < MAX_SCATTER;
  const bool condition = tolx_condition && toly_condition && scatter_condition;

  return condition * scatter + !condition * MAX_FLOAT;
}

/**
 * @brief Fills dev_hit_candidates.
 * 
 * @param dev_hit_candidates 
 * @param no_sensors         
 * @param sensor_hitStarts   
 * @param sensor_hitNums     
 * @param hit_Xs             
 * @param hit_Ys             
 * @param hit_Zs             
 */
__device__ void fillCandidates(int* const hit_candidates, const int no_sensors, 
  const int* const sensor_hitStarts, const int* const sensor_hitNums,
  const float* const hit_Xs, const float* const hit_Ys, const float* const hit_Zs) {

  const int blockDim_product = blockDim.x * blockDim.y;
  int first_sensor = 51;
  while (first_sensor >= 4) {
    const int second_sensor = first_sensor - 2;
    int hit_shift = 1;

    // Optional: Do it with z from sensors
    // zs of both sensors
    // const float z_first_sensor = ... ;

    // Iterate in all hits in z0
    for (int i=0; i<((int) ceilf( ((float) sensor_hitNums[first_sensor]) / blockDim_product)); ++i) {
      const int h0_element = blockDim.x * i + threadIdx.x * blockDim.y + threadIdx.y;
      bool inside_bounds = h0_element < sensor_hitNums[first_sensor];

      if (inside_bounds) {
        const int h0_index = sensor_hitStarts[first_sensor] + h0_element;
        Hit h0 {hit_Xs[h0_index], hit_Ys[h0_index], hit_Zs[h0_index]};
        
        // Iterate in all hits in z1
        for (int h1_element=0; h1_element<sensor_hitNums[second_sensor]; ++h1_element) {
          inside_bounds = h1_element < sensor_hitNums[second_sensor];

          if (inside_bounds) {
            const int h1_index = sensor_hitStarts[second_sensor] + h1_element;
            Hit h1 {hit_Xs[h1_index], hit_Ys[h1_index], hit_Zs[h1_index]};

            // Check if h0 and h1 are compatible
            const float h_dist = fabs(h1.z - h0.z);
            const float dxmax = PARAM_MAXXSLOPE * h_dist;
            const float dymax = PARAM_MAXYSLOPE * h_dist;
            if (fabs(h1.x - h0.x) < dxmax && fabs(h1.y - h0.y) < dymax) {
              dev_hit_candidates[h0_index * NUM_MAX_CANDIDATES + hit_shift++] = h1_index;
            }

            if (hit_shift == NUM_MAX_CANDIDATES) {
                // Ugly - Check if this happens
                // NUM_MAX_CANDIDATES has to be higher, or it has to
                // grow dinamically
                break;
            }
          }
        }

        // The first element contains how many compatible hits are there
        dev_hit_candidates[h0_index * NUM_MAX_CANDIDATES] = hit_shift - 1;
      }
    }

    --first_sensor;
  }
}
