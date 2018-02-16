#include "SearchByTriplet.cuh"

__device__ void weakTracksAdder(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  bool* hit_used
) {
  // Compute the weak tracks
  const auto weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim.x - 1) / blockDim.x; ++i) {
    const auto weaktrack_no = blockDim.x * i + threadIdx.x;
    if (weaktrack_no < weaktracks_total) {
      // Load the tracks from the tracklets
      const Track t = tracklets[weak_tracks[weaktrack_no]];
      const unsigned int used = hit_used[t.hits[0]] + hit_used[t.hits[1]] + hit_used[t.hits[2]];

      // Store them in the tracks bag
      if (used < 1) {
        const unsigned int trackno = atomicAdd(tracks_insertPointer, 1);
        ASSERT(trackno < MAX_TRACKS)
        tracks[trackno] = t;
      }
    }
  }
}

/**
 * @brief Uses shared memory to check used hits
 *        in a range of weak tracks.
 *        
 *        Since weak tracks order is not guaranteed,
 *        this method may produce a varying number of tracks
 */
__device__ void weakTracksAdderShared(
  int* shared_hits,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  bool* hit_used
) {
  const auto weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim.x - 1) / blockDim.x; ++i) {
    const auto weaktrack_no = blockDim.x * i + threadIdx.x;
    // Shared checking mechanism
    shared_hits[threadIdx.x] = -1;
    if (weaktrack_no < weaktracks_total) {
      const Track t = tracklets[weak_tracks[weaktrack_no]];
      bool clone = hit_used[t.hits[0]] || hit_used[t.hits[1]] && hit_used[t.hits[2]];
      if (!clone) {
        for (int h=0; h<3; ++h) {
          shared_hits[threadIdx.x] = t.hits[h];
          __syncthreads();
          for (int shared_id=threadIdx.x+1; shared_id<blockDim.x; ++shared_id) {
            const auto other_hit = shared_hits[shared_id];
            clone |= other_hit==t.hits[0] || other_hit==t.hits[1] || other_hit==t.hits[2];
          }
          __syncthreads();
        }
      }

      // Check again
      if (!clone) {
        // Store them in the tracks bag
        // Note: They have already been checked for used hits
        const unsigned int trackno = atomicAdd(tracks_insertPointer, 1);
        ASSERT(trackno < MAX_TRACKS)
        tracks[trackno] = t;
      }
    }
  }
}
