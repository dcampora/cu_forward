#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#define MAX_TRACKS 8000
#define MAX_TRACK_SIZE 24

#define BUNCH_POST_TRACKS 32
#define HITS_SHARED 32
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

struct Sensor {
	int hitStart;
	int hitNums;
};

struct Hit {
	float x;
	float y;
	float z;
};

struct Track { // 4 + 24 * 4 = 100 B
	// float x0, tx, y0, ty; // deprecated
	
	int hitsNum;
	int hits[MAX_TRACK_SIZE];
};

#endif
