#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#define NUMTHREADS_X 64
#define NUM_ATOMICS 4

#define MAX_TRACKS 8000
#define MAX_TRACK_SIZE 24

#define REQUIRED_UNIQUES 0.6f
#define MIN_HITS_TRACK 3
#define MAX_FLOAT 100000000.0
#define MAX_SKIPPED_MODULES 3

#define PARAM_W 3966.94f // 0.050 / sqrt( 12. )
#define PARAM_MAXXSLOPE 0.4f
#define PARAM_MAXYSLOPE 0.3f

#define PARAM_TOLERANCE_BASIC 0.15f
#define PARAM_TOLERANCE_EXTENDED 0.3f
#define PARAM_TOLERANCE_EXTRA 0.6f

#define PARAM_TOLERANCE PARAM_TOLERANCE_EXTRA

#define MAX_SCATTER 0.000016f
#define SENSOR_DATA_HITNUMS 3

struct Sensor {
	unsigned int hitStart;
	unsigned int hitNums;

    __device__ Sensor(){}
    __device__ Sensor(const int _hitStart, const int _hitNums) : 
        hitStart(_hitStart), hitNums(_hitNums) {}
};

struct Hit {
	float x;
	float y;
	float z;

    __device__ Hit(){}
    __device__ Hit(const float _x, const float _y, const float _z) :
        x(_x), y(_y), z(_z) {}
};

struct Track { // 4 + 24 * 4 = 100 B
	// float x0, tx, y0, ty; // deprecated
	unsigned int hitsNum;
	unsigned int hits[MAX_TRACK_SIZE];
};

struct Tracklet {
    unsigned int hitsNum;
    unsigned int h0, h1, h2;

    __device__ Tracklet(){}
    __device__ Tracklet(const unsigned int _hitsNum, unsigned int _h0, unsigned int _h1, unsigned int _h2) : 
        hitsNum(_hitsNum), h0(_h0), h1(_h1), h2(_h2) {}
};

#endif
