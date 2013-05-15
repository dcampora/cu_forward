
#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

struct Track { // 57 + 25*4 = 328 B
	float m_x0;
	float m_tx;
	float m_y0;
	float m_ty;

	float m_s0;
	float m_sx;
	float m_sz;
	float m_sxz;
	float m_sz2;

	float m_u0;
	float m_uy;
	float m_uz;
	float m_uyz;
	float m_uz2;
	
	char trackHitsNum;
	int hits[25];
};

__device__ int max_hits;

__device__ __constant__ int sens_num = 48;
__device__ int hits_num;

__device__ int* no_sensors;
__device__ int* no_hits;
__device__ int* sensor_Zs;
__device__ int* sensor_hitStarts;
__device__ int* sensor_hitNums;
__device__ int* hit_IDs;
__device__ double* hit_Xs;
__device__ double* hit_Ys;
__device__ int* hit_Zs;

__device__ __constant__ float 	f_m_maxXSlope			= 0.4;
__device__ __constant__ float 	f_m_maxYSlope			= 0.3;
__device__ __constant__ float 	f_m_maxZForRBeamCut		= 200.0;
__device__ __constant__ float 	f_m_maxR2Beam			= 1.0;
__device__ __constant__ int 	f_m_maxMissed			= 4;
__device__ __constant__ float 	f_m_extraTol			= 0.150;
__device__ __constant__ float 	f_m_maxChi2ToAdd		= 100.0;
__device__ __constant__ float 	f_m_maxChi2SameSensor	= 16.0;
__device__ __constant__ float   f_m_maxChi2Short		= 6.0 ;
__device__ __constant__ float   f_m_maxChi2PerHit		= 16.0;
__device__ __constant__ int 	f_m_sensNum				= 48;
__device__ __constant__ float   f_w						= 0.0144338; // 0.050 / sqrt( 12. )

#endif
