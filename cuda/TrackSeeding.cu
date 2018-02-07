#include "SearchByTriplet.cuh"

/**
 * @brief Track Seeding
 * 
 * @param hit_Xs                  
 * @param hit_Ys                  
 * @param hit_Zs                  
 * @param sensor_data             
 * @param hit_candidates          
 * @param max_numhits_to_process  
 * @param sh_hit_x                
 * @param sh_hit_y                
 * @param sh_hit_z                
 * @param sh_hit_process          
 * @param hit_used                
 * @param hit_h2_candidates       
 * @param blockDim_sh_hit         
 * @param best_fits               
 * @param tracklets_insertPointer 
 * @param ttf_insertPointer       
 * @param tracklets               
 * @param tracks_to_follow        
 */
__device__ void trackSeeding(
#if USE_SHARED_FOR_HITS
  float* const sh_hit_x,
  float* const sh_hit_y,
  float* const sh_hit_z,
#endif
  const float* const hit_Xs,
  const float* const hit_Ys,
  const float* const hit_Zs,
  int* const sensor_data,
  int* const hit_candidates,
  unsigned int* const max_numhits_to_process,
  int* const sh_hit_process,
  bool* const hit_used,
  int* const hit_h2_candidates,
  const int blockDim_sh_hit,
  float* const best_fits,
  unsigned int* const tracklets_insertPointer,
  unsigned int* const ttf_insertPointer,
  Track* const tracklets,
  int* const tracks_to_follow
) {

  // Track creation starts
  unsigned int best_hit_h1, best_hit_h2;
  Hit h0, h1;
  int first_h1, first_h2, last_h2;
  float dymax;

  const int h0_index = sh_hit_process[threadIdx.x];
  bool inside_bounds = h0_index != -1;
  unsigned int num_h1_to_process = 0;
  float best_fit = MAX_FLOAT;

  // We will repeat this for performance reasons
  if (inside_bounds) {
    h0.x = hit_Xs[h0_index];
    h0.y = hit_Ys[h0_index];
    h0.z = hit_Zs[h0_index];
    
    // Calculate new dymax
    const float s1_z = hit_Zs[sensor_data[1]];
    const float h_dist = fabs(s1_z - h0.z);
    dymax = PARAM_MAXYSLOPE * h_dist;

    // Only iterate in the hits indicated by hit_candidates :)
    first_h1 = hit_candidates[2 * h0_index];
    const int last_h1 = hit_candidates[2 * h0_index + 1];
    num_h1_to_process = last_h1 - first_h1;
    atomicMax(max_numhits_to_process, num_h1_to_process);
    ASSERT(max_numhits_to_process[0] >= num_h1_to_process)
  }

  __syncthreads();

  // Only iterate max_numhits_to_process[0] iterations (with blockDim.y threads) :D :D :D
  for (int j=0; j<(max_numhits_to_process[0] + blockDim.y - 1) / blockDim.y; ++j) {
    const int h1_element = blockDim.y * j + threadIdx.y;
    inside_bounds &= h1_element < num_h1_to_process; // Hmmm...
    bool is_h1_used = true; // TODO: Can be merged with h1_element restriction
    int h1_index;
    float dz_inverted;

    if (inside_bounds) {
      h1_index = first_h1 + h1_element;
      is_h1_used = hit_used[h1_index];
      if (!is_h1_used) {
        h1.x = hit_Xs[h1_index];
        h1.y = hit_Ys[h1_index];
        h1.z = hit_Zs[h1_index];

        dz_inverted = 1.f / (h1.z - h0.z);
      }

      first_h2 = hit_h2_candidates[2 * h1_index];
      last_h2 = hit_h2_candidates[2 * h1_index + 1];
      // In case there be no h2 to process,
      // we can preemptively prevent further processing
      inside_bounds &= first_h2 != -1;
    }

    // Iterate in the third list of hits
    // Tiled memory access on h2
    for (int k=0; k<(sensor_data[SENSOR_DATA_HITNUMS + 2] + blockDim_sh_hit - 1) / blockDim_sh_hit; ++k) {

#if USE_SHARED_FOR_HITS
      __syncthreads();
      if (threadIdx.y < SH_HIT_MULT) {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int sh_hit_no = blockDim_sh_hit * k + tid;
        if (sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2]) {
          const int h2_index = sensor_data[2] + sh_hit_no;

          // Coalesced memory accesses
          ASSERT(tid < blockDim_sh_hit)
          sh_hit_x[tid] = hit_Xs[h2_index];
          sh_hit_y[tid] = hit_Ys[h2_index];
          sh_hit_z[tid] = hit_Zs[h2_index];
        }
      }
      __syncthreads();
#endif

      if (inside_bounds && !is_h1_used) {

        const int last_hit_h2 = min(blockDim_sh_hit * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
        for (int kk=blockDim_sh_hit * k; kk<last_hit_h2; ++kk) {

          const int h2_index = sensor_data[2] + kk;
          if (h2_index >= first_h2 && h2_index < last_h2) {
#if USE_SHARED_FOR_HITS
            const int sh_h2_index = kk % blockDim_sh_hit;
            const Hit h2 {sh_hit_x[sh_h2_index], sh_hit_y[sh_h2_index], sh_hit_z[sh_h2_index]};
#else
            const Hit h2 {hit_Xs[h2_index], hit_Ys[h2_index], hit_Zs[h2_index]};
#endif

            // Predictions of x and y for this hit
            const float z2_tz = (h2.z - h0.z) * dz_inverted;
            const float x = h0.x + (h1.x - h0.x) * z2_tz;
            const float y = h0.y + (h1.y - h0.y) * z2_tz;
            const float dx = x - h2.x;
            const float dy = y - h2.y;

            if (fabs(h1.y - h0.y) < dymax && fabs(dx) < PARAM_TOLERANCE && fabs(dy) < PARAM_TOLERANCE) {
              // Calculate fit
              const float scatterNum = (dx * dx) + (dy * dy);
              const float scatterDenom = 1.f / (h2.z - h1.z);
              const float scatter = scatterNum * scatterDenom * scatterDenom;
              const bool condition = scatter < MAX_SCATTER;
              const float fit = condition * scatter + !condition * MAX_FLOAT; 

              const bool fit_is_better = fit < best_fit;
              best_fit = fit_is_better * fit + !fit_is_better * best_fit;
              best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
              best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
            }
          }
        }
      }
    }
  }

  // Compare / Mix the results from the blockDim.y threads
  ASSERT(threadIdx.x * blockDim.y + threadIdx.y < blockDim.x * MAX_NUMTHREADS_Y)
  best_fits[threadIdx.x * blockDim.y + threadIdx.y] = best_fit;

  __syncthreads();

  bool accept_track = false;
  if (h0_index != -1 && best_fit != MAX_FLOAT) {
    best_fit = MAX_FLOAT;
    int threadIdx_y_winner = -1;
    for (int i=0; i<blockDim.y; ++i) {
      const float fit = best_fits[threadIdx.x * blockDim.y + i];
      if (fit < best_fit) {
        best_fit = fit;
        threadIdx_y_winner = i;
      }
    }
    accept_track = threadIdx.y == threadIdx_y_winner;
  }

  // We have a best fit! - haven't we?
  // Only go through the tracks on the selected thread
  if (accept_track) {
    // Fill in track information

    // Add the track to the bag of tracks
    const unsigned int trackP = atomicAdd(tracklets_insertPointer, 1);
    // ASSERT(trackP < number_of_hits)
    tracklets[trackP] = Track {3, (unsigned int) sh_hit_process[threadIdx.x], best_hit_h1, best_hit_h2};;

    // Add the tracks to the bag of tracks to_follow
    // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
    // and hence it is stored in tracklets
    const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1) % TTF_MODULO;
    tracks_to_follow[ttfP] = 0x80000000 | trackP;
  }
}