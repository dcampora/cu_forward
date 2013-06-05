
#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define MAX_TRACKS 10000
#define TRACK_SIZE 24

struct Sensor {
	int z;
	int hitStart;
	int hitNums;
};

struct Hit {
	float x;
	float y;
};

struct Track { // 57 + 24*4 = 324 B
	float x0;
	float tx;
	float y0;
	float ty;

	float s0;
	float sx;
	float sz;
	float sxz;
	float sz2;

	float u0;
	float uy;
	float uz;
	float uyz;
	float uz2;
	
	char hitsNum;
	int hits[TRACK_SIZE];
};

// __device__ int max_hits;
// __device__ int hits_num;

__device__ __constant__ int sens_num = 48;

__device__ int* no_sensors;
__device__ int* no_hits;
__device__ int* sensor_Zs;
__device__ int* sensor_hitStarts;
__device__ int* sensor_hitNums;
__device__ int* hit_IDs;
__device__ float* hit_Xs;
__device__ float* hit_Ys;
__device__ int* hit_Zs;

__device__ int* prevs;
__device__ int* nexts;

#define PARAM_W 0.0144338f // 0.050 / sqrt( 12. )
#define PARAM_MAXXSLOPE 0.4f
#define PARAM_MAXYSLOPE 0.3f

#define PARAM_TOLERANCE 0.15f
#define PARAM_TOLERANCE_EXTENDED 0.3f

#define PARAM_MAXCHI2 100.0f
#define PARAM_MAXCHI2_EXTENDED 200.0f

/*
__device__ __constant__ float 	f_m_maxXSlope			= 0.4f;
__device__ __constant__ float 	f_m_maxYSlope			= 0.3f;
__device__ __constant__ float 	f_m_maxZForRBeamCut		= 200.0f;
__device__ __constant__ float 	f_m_maxR2Beam			= 1.0f;
__device__ __constant__ int 	f_m_maxMissed			= 4;
__device__ __constant__ float 	f_m_extraTol			= 0.150f;
__device__ __constant__ float 	f_m_maxChi2ToAdd		= 100.0f;
__device__ __constant__ float 	f_m_maxChi2SameSensor	= 16.0f;
__device__ __constant__ float   f_m_maxChi2Short		= 6.0f;
__device__ __constant__ float   f_m_maxChi2PerHit		= 16.0f;
__device__ __constant__ int 	f_m_sensNum				= 48;
__device__ __constant__ float   f_w						= 0.0144338f; // 0.050 / sqrt( 12. )
*/

#endif
