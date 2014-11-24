#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#define MAX_TRACKS 10000
#define MAX_TRACK_SIZE 24

#define ALLOW_POSTPROCESSING 0
#define BUNCH_POST_TRACKS 32
#define REQUIRED_UNIQUES 0.6f
#define MIN_HITS_TRACK 3

#define PARAM_W 3966.94f // 0.050 / sqrt( 12. )
#define PARAM_MAXXSLOPE 0.4f
#define PARAM_MAXYSLOPE 0.3f

#define PARAM_TOLERANCE_BASIC 0.15f
#define PARAM_TOLERANCE_EXTENDED 0.3f

#define PARAM_MAXCHI2_BASIC 100.0f
#define PARAM_MAXCHI2_EXTENDED 200.0f

#define PARAM_TOLERANCE PARAM_TOLERANCE_EXTENDED
#define PARAM_MAXCHI2 PARAM_MAXCHI2_EXTENDED

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
	
	int hitsNum;
	int hits[MAX_TRACK_SIZE];
};

struct TrackFit {
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
};

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
