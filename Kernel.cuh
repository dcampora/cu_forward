#ifndef CUDA_KERNEL
#define CUDA_KERNEL 1

#include "Definitions.cuh"

#define HITS_SHARED 32
#define MAX_FLOAT 100000000.0
#define NUM_SENSORS 48
#define POST_PROCESSING 0

/*
float   f_m_maxXSlope           = 0.400;
float   f_m_maxYSlope           = 0.300;
float   f_m_maxZForRBeamCut     = 200.0;
float   f_m_maxR2Beam           = 1.0;
int     f_m_maxMissed           = 4;
float   f_m_extraTol            = 0.150;
float   f_m_maxChi2ToAdd        = 100.0;
float   f_m_maxChi2SameSensor   = 16.0;
float   f_m_maxChi2Short        = 6.0 ;
float   f_m_maxChi2PerHit       = 16.0;
int     f_m_sensNum             = 48;
float   f_w                     = 0.050 / sqrt( 12. );
*/

__device__ float fitHits(Hit& h0, Hit& h1, Sensor& s0, Sensor& s1);
__device__ float fitHitToTrack(Track& t, Hit& h1, Sensor& s1);
__device__ void acceptTrack(Track& t, TrackFit& fit, Hit& h0, Hit& h1, Sensor& s0, Sensor& s1, int h0_num, int h1_num);
__device__ void updateTrack(Track& t, TrackFit& fit, Hit& h1, Sensor& s1, int h1_num);
__device__ void updateTrackCoords(Track& t, TrackFit& fit);
__device__ float trackChi2(Track& t);
__device__ float hitChi2(Track& t, Hit& h, int hit_z);

__global__ void prepareData(char* input, int* _prevs, int* _nexts, bool* track_holders);
__global__ void neighboursFinder();
__global__ void neighboursCleaner();

__global__ void gpuKalman(Track* tracks, bool* track_holders);
__global__ void postProcess(Track* tracks, bool* track_holders, int* track_indexes, int* num_tracks, int* tracks_to_process);


// __device__ int max_hits;
// __device__ int hits_num;

#endif
