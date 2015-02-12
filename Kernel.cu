#include "Kernel.cuh"

// __device__ __constant__ int sens_num = 48;

__device__ int* no_sensors;
__device__ int* no_hits;
__device__ int* sensor_Zs;
__device__ int* sensor_hitStarts;
__device__ int* sensor_hitNums;
__device__ unsigned int* hit_IDs;
__device__ float* hit_Xs;
__device__ float* hit_Ys;
__device__ float* hit_Zs;

__device__ bool* hit_used;
__device__ int* prevs;
__device__ int* nexts;

__device__ int* tracks_to_follow_q1;
__device__ int* tracks_to_follow_q2;
__device__ int* atomicsStorage;
__device__ unsigned int* ttf_insertPointer;
__device__ unsigned int* tracks_insertPointer;
__device__ unsigned int* weaktracks_insertPointer;
__device__ unsigned int* tracklets_insertPointer;

__device__ Track* tracklets;
__device__ int* weak_tracks; // Tracks with only three hits


__global__ void prepareData(char* input, int* _tracks_to_follow_q1, int* _tracks_to_follow_q2,
    bool* used, int* _atomicsStorage, Track* dev_tracklets, int* dev_weak_tracks) {
  no_sensors = (int*) &input[0];
  no_hits = (int*) (no_sensors + 1);
  sensor_Zs = (int*) (no_hits + 1);
  sensor_hitStarts = (int*) (sensor_Zs + no_sensors[0]);
  sensor_hitNums = (int*) (sensor_hitStarts + no_sensors[0]);
  hit_IDs = (unsigned int*) (sensor_hitNums + no_sensors[0]);
  hit_Xs = (float*) (hit_IDs + no_hits[0]);
  hit_Ys = (float*) (hit_Xs + no_hits[0]);
  hit_Zs = (float*) (hit_Ys + no_hits[0]);

  hit_used = used;
  tracks_to_follow_q1 = _tracks_to_follow_q1;
  tracks_to_follow_q2 = _tracks_to_follow_q2;

  tracklets = dev_tracklets;
  weak_tracks = dev_weak_tracks;
  
  atomicsStorage = _atomicsStorage;
  ttf_insertPointer = (unsigned int*) &atomicsStorage[0];
  tracks_insertPointer = (unsigned int*) &atomicsStorage[1];
  weaktracks_insertPointer = (unsigned int*) &atomicsStorage[2];
  tracklets_insertPointer = (unsigned int*) &atomicsStorage[3];

  // TODO: We can do a calloc or memcpy a calloc
  for (int i=0; i<*no_hits; ++i){
    hit_used[i] = false;
  }

  for (int i=0; i<10; ++i){
    atomicsStorage[i] = 0;
  }
}

/** fitHits, gives the fit between h0 and h1.

The accept condition requires dxmax and dymax to be in a range.

The fit (d1) depends on the distance of the tracklet to <0,0,0>.
*/
__device__ float fitHits(Hit& h0, Hit& h1, Hit &h2) {
  // Max dx, dy permissible over next hit

  // TODO: This can go outside this function (only calc once per pair
  // of sensors). Also, it could only be calculated on best fitting distance d1.
  const float h_dist = fabs((float)( h1.z - h0.z ));
  float dxmax = PARAM_MAXXSLOPE * h_dist;
  float dymax = PARAM_MAXYSLOPE * h_dist;
  
  bool accept_condition = fabs(h1.x - h0.x) < dxmax && fabs(h1.y - h0.y) < dymax;

  // First approximation -
  // With the sensor z, instead of the hit z
  float z2_tz = ((float) h2.z - h0.z) / ((float) (h1.z - h0.z));
  float x = h0.x + (h1.x - h0.x) * z2_tz;
  float y = h0.y + (h1.y - h0.y) * z2_tz;

  float dx = x - h2.x;
  float dy = y - h2.y;
  float chi2 = dx * dx * PARAM_W + dy * dy * PARAM_W;
  // accept_condition &= chi2 < PARAM_MAXCHI2; // No need for this

  return accept_condition * chi2 + !accept_condition * MAX_FLOAT;
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
 * @param h2 
 * @return 
 */
__device__ float fitHitToTrack(const float tx, const float ty, const Hit& h0, const Hit& h2){
  // tolerances
  const float dz = h2.z - h0.z;
  const float x_prediction = h0.x + tx * dz;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < PARAM_TOLERANCE;

  const float y_prediction = h0.y + ty * dz;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < PARAM_TOLERANCE;

  // chi2 - how good is this fit
  const float chi2 = dx * dx * PARAM_W + dy * dy * PARAM_W;
  const bool condition = tolx_condition && toly_condition;

  return condition * chi2 + !condition * MAX_FLOAT;
}

/** Simple implementation of the Kalman Filter selection on the GPU (step 4).

Will rely on pre-processing for selecting next-hits for each hit.

Implementation,
- Perform implementation searching on all hits for each sensor

The algorithm has two parts:
- Track creation (two hits)
- Track following (consecutive sensors)


Optimizations,
- Optimize with shared memory
- Optimize further with pre-processing

Then there must be a post-processing, which selects the
best tracks based on (as per the conversation with David):
- length
- chi2

For this, simply use the table with all created tracks (postProcess):

#track, h0, h1, h2, h3, ..., hn, length, chi2

*/

__global__ void searchByTriplet(Track* tracks) {
  Track t;
  Sensor s0, s1, s2;
  Hit h0, h1, h2;

  float fit, best_fit;
  bool fit_is_better, accept_track;
  int best_hit_h1, best_hit_h2;

  int* tracks_to_follow      = tracks_to_follow_q1;
  int* prev_tracks_to_follow = tracks_to_follow_q2;
  int* temp_tracks_to_follow;

  for (int i=0; i<2; ++i){
    // Deal with odd or even separately
    int first_sensor = 51 - i;

    // Prepare s1 and s2 for the first iteration
    const int second_sensor = first_sensor - 2;

    s1.hitStart = sensor_hitStarts[first_sensor];
    s1.hitNums = sensor_hitNums[first_sensor];
    s2.hitStart = sensor_hitStarts[second_sensor];
    s2.hitNums = sensor_hitNums[second_sensor];

    while (first_sensor >= 4) {
      // Iterate in sensors
      // Reuse the info from last sensors
      s0 = s1;
      s1 = s2;

      const int third_sensor = first_sensor - 4;
      s2.hitStart = sensor_hitStarts[third_sensor];
      s2.hitNums = sensor_hitNums[third_sensor];

      // Exchange track_to_follow's
      temp_tracks_to_follow = prev_tracks_to_follow;
      prev_tracks_to_follow = tracks_to_follow;
      tracks_to_follow = temp_tracks_to_follow;

      // Reset the ttf_insertPointer and synchronize. It's a bit sad we
      // have to synchronize twice.
      // TODO: This is ugly
      const unsigned int last_ttf_insertPointer = ttf_insertPointer[0];
      __syncthreads();
      if (threadIdx.x == 0)
        ttf_insertPointer[0] = 0;
      __syncthreads();

      // 2a. Track following
      for (int i=0; i<int(ceilf( ((float) last_ttf_insertPointer) / blockDim.x)); ++i) {
        const int ttf_element = blockDim.x * i + threadIdx.x;

        if (ttf_element < last_ttf_insertPointer) {
          int trackno = prev_tracks_to_follow[ttf_element];

          const Track* track_pointer = (trackno & 0x80000000) == 0x80000000 ? tracklets : tracks;
          t = track_pointer[trackno & 0x7FFFFFFF];

          // Load last two hits in h0, h1
          const int t_hitsNum = t.hitsNum;
          const int h0_num = t.hits[t_hitsNum - 2];
          const int h1_num = t.hits[t_hitsNum - 1];

          h0.x = hit_Xs[h0_num];
          h0.y = hit_Ys[h0_num];
          h0.z = hit_Zs[h0_num];

          h1.x = hit_Xs[h1_num];
          h1.y = hit_Ys[h1_num];
          h1.z = hit_Zs[h1_num];

          // Track following over t, for all hits in the next module
          // Line calculations
          const float td = 1.0f / (h1.z - h0.z);
          const float txn = (h1.x - h0.x);
          const float tyn = (h1.y - h0.y);
          const float tx = txn * td;
          const float ty = tyn * td;

          // Find a best fit
          best_fit = MAX_FLOAT;
          for (int k=0; k<s2.hitNums; ++k) {
            const int h2_index = s2.hitStart + k;
            h2.x = hit_Xs[h2_index];
            h2.y = hit_Ys[h2_index];
            h2.z = hit_Zs[h2_index];

            fit = fitHitToTrack(tx, ty, h0, h2);
            fit_is_better = fit < best_fit;

            best_fit = fit_is_better * fit + !fit_is_better * best_fit;
            best_hit_h2 = fit_is_better * h2_index + !fit_is_better * best_hit_h2;
          }

          // We have a best fit!
          // Fill in t, ONLY in case the best fit is acceptable
          if (best_fit != MAX_FLOAT) {
            // Reload h2
            h2.x = hit_Xs[best_hit_h2];
            h2.y = hit_Ys[best_hit_h2];
            h2.z = hit_Zs[best_hit_h2];

            // Mark h2 as used
            hit_used[best_hit_h2] = true;
            
            // Update the tracks to follow, we'll have to follow up
            // this track on the next iteration :)
            // updateTrack(t, tfit, h2, best_hit_h2);
            t.hits[t.hitsNum++] = best_hit_h2;

            // Update the track in the bag
            if (t.hitsNum > 4){
              // If it is a track made out of *strictly* more than four hits,
              // the trackno refers to the tracks location.
              tracks[trackno] = t;
            }
            else {
              // Otherwise, we have to allocate it in the tracks,
              // and update trackno
              trackno = atomicAdd(tracks_insertPointer, 1);
              tracks[trackno] = t;
            }

            // Add the tracks to the bag of tracks to_follow
            const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
            tracks_to_follow[ttfP] = trackno;
          }
          // In the "else" case, we couldn't follow up the track,
          // so we won't be track following it anymore.
          
          else if (t.hitsNum == 3){
            // If there are only three hits in this track,
            // mark it as "doubtful"
            const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
            weak_tracks[weakP] = trackno & 0x7FFFFFFF;
          }

          else {
            // There are more than three hits in this track,
            // but we didn't find any further hits.
            // Mark the first three hits as used. (all the other
            // are already marked :) )
            hit_used[t.hits[0]] = true;
            hit_used[t.hits[1]] = true;
            hit_used[t.hits[2]] = true;
          }
        }
      }

      // Iterate in all hits for current sensor
      // 2a. Seeding
      for (int i=0; i<int(ceilf( ((float) s0.hitNums) / blockDim.x)); ++i) {
        const int first_hit = blockDim.x * i + threadIdx.x;
        const int h0_index = s0.hitStart + first_hit;
        const bool is_used = hit_used[h0_index];

        if (!is_used && first_hit < s0.hitNums) {
          h0.x = hit_Xs[h0_index];
          h0.y = hit_Ys[h0_index];
          h0.z = hit_Zs[h0_index];

          // TODO: This is actually unnecessary, just a sanity check
          // Initialize track
          for(int j=0; j<MAX_TRACK_SIZE; ++j) {
            t.hits[j] = -1;
          }
      
          // TRACK CREATION
          best_fit = MAX_FLOAT;
          // best_hit_h1 = -1;
          // best_hit_h2 = -1;
          for (int j=0; j<s1.hitNums; ++j) {
            const int h1_index = s1.hitStart + j;
            h1.x = hit_Xs[h1_index];
            h1.y = hit_Ys[h1_index];
            h1.z = hit_Zs[h1_index];

            // Iterate in the third! list of hits
            for (int k=0; k<s2.hitNums; ++k) {
              const int h2_index = s2.hitStart + k;
              h2.x = hit_Xs[h2_index];
              h2.y = hit_Ys[h2_index];
              h2.z = hit_Zs[h2_index];

              fit = fitHits(h0, h1, h2);
              fit_is_better = fit < best_fit;

              best_fit = fit_is_better * fit + !fit_is_better * best_fit;
              best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
              best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
            }
          }

          // We have a best fit! - haven't we?
          accept_track = best_fit != MAX_FLOAT;

          if (accept_track) {
            // Reload h1 and h2
            h1.x = hit_Xs[best_hit_h1];
            h1.y = hit_Ys[best_hit_h1];
            h1.z = hit_Zs[best_hit_h1];

            h2.x = hit_Xs[best_hit_h2];
            h2.y = hit_Ys[best_hit_h2];
            h2.z = hit_Zs[best_hit_h2];

            // Fill in track information
            // acceptTrack(t, tfit, h0, h1, s0.hitStart + first_hit, best_hit_h1);
            // updateTrack(t, tfit, h2, best_hit_h2);
            t.hitsNum = 3;
            t.hits[0] = s0.hitStart + first_hit;
            t.hits[1] = best_hit_h1;
            t.hits[2] = best_hit_h2;

            // Add the track to the bag of tracks
            const unsigned int trackP = atomicAdd(tracklets_insertPointer, 1);
            tracklets[trackP] = t;

            // Add the tracks to the bag of tracks to_follow
            // Note: The first bit marks if this is a tracklet or a full track (>=4 hits)
            const unsigned int ttfP = atomicAdd(ttf_insertPointer, 1);
            tracks_to_follow[ttfP] = 0x80000000 | trackP;
          }
        }
      }

      first_sensor -= 2;
    }

    // Process the last bunch of track_to_follows
    const unsigned int last_ttf_insertPointer = ttf_insertPointer[0];
    for (int i=0; i<int(ceilf( ((float) last_ttf_insertPointer) / blockDim.x)); ++i) {
      const int ttf_element = blockDim.x * i + threadIdx.x;

      if (ttf_element < last_ttf_insertPointer) {
        const int trackno = tracks_to_follow[ttf_element];

        const Track* track_pointer = (trackno & 0x80000000) == 0x80000000 ? tracklets : tracks;
        t = track_pointer[trackno & 0x7FFFFFFF];

        if (t.hitsNum == 3){
          // If there are only three hits in this track,
          // mark it as "doubtful"
          const unsigned int weakP = atomicAdd(weaktracks_insertPointer, 1);
          weak_tracks[weakP] = trackno & 0x7FFFFFFF;
        }

        else {
          // There are more than three hits in this track.
          // Mark the first three hits as used. (all the other
          // are already marked :) )
          hit_used[t.hits[0]] = true;
          hit_used[t.hits[1]] = true;
          hit_used[t.hits[2]] = true;
        }
      }
    }

    // Compute the three-hit tracks left
    const int weaktracks_total = weaktracks_insertPointer[0];
    for (int i=0; i<int(ceilf( ((float) weaktracks_total) / blockDim.x)); ++i) {
      const int weaktrack_no = blockDim.x * i + threadIdx.x;
      if (weaktrack_no < weaktracks_total){
        // Load the tracks from the tracklets
        t = tracklets[weak_tracks[weaktrack_no]];

        // Store them in the tracks bag iff they
        // are made out of three unused hits
        if (!hit_used[t.hits[0]] &&
            !hit_used[t.hits[1]] &&
            !hit_used[t.hits[2]]){

          const int trackno = atomicAdd(tracks_insertPointer, 1);
          tracks[trackno] = t;
        }
      }
    }
  }
}
