
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#include "Definitions.cuh"

__device__ float f_zBeam(track *tr);
__device__ float f_r2AtZ( float z , track *tr);
__device__ void f_solve (track *tr);
__device__ float f_chi2Hit( float x, float y, float hitX, float hitY, float hitW);
__device__ float f_xAtHit(track *tr, float z );
__device__ float f_yAtHit( track *tr, float z  );
__device__ float f_chi2Track(track *tr, int offset);
__device__ float f_chi2(track *t);
__device__ void f_removeHit(track *t, int worstHitOffset);
__device__ void f_removeWorstHit(track* t);
__device__ bool f_all3SensorsAreDifferent(track t);
__device__ int f_nbUnused(track t);
__device__ void f_addHit ( track *tr, int offset, int lastZ);
__device__ void f_setTrack(track *tr, int hit0offset, int hit1offset, int lastZ);
