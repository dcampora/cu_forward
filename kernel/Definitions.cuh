#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

// Number of threads
#define NUMTHREADS_X 64

// How many concurrent h1s to process max
// It should be a divisor of NUMTHREADS_X
#define MAX_CONCURRENT_H1 16

#define NUM_ATOMICS 5
#define USE_SHARED_FOR_HITS false
#define SH_HIT_MULT 2

#define MAX_TRACKS 3000
#define MAX_TRACK_SIZE 24

#define REQUIRED_UNIQUES 0.6f
#define MIN_HITS_TRACK 3
#define MAX_FLOAT FLT_MAX
#define MIN_FLOAT -FLT_MAX
#define MAX_SKIPPED_MODULES 1
#define TTF_MODULO 2000
#define MAX_NUMHITS_IN_MODULE 300
#define PARAM_W 3966.94f // 0.050 / sqrt( 12. )

// These parameters heavily impact the found tracks
#define PARAM_TOLERANCE_ALPHA 0.2f
#define PARAM_TOLERANCE_BETA 0.1f
#define PARAM_TOLERANCE 0.6f
#define MAX_SCATTER 0.000016f

#define MODULE_DATA_HITNUMS 3
#define RESULTS_FOLDER "results"

#define PRINT_SOLUTION false
#define PRINT_FILL_CANDIDATES false
#define PRINT_VERBOSE false
#define PRINT_BINARY false
#define ASSERTS_ENABLED false

#if ASSERTS_ENABLED == true
#include "assert.h"
#define ASSERT(EXPR) assert(EXPR);
#else
#define ASSERT(EXPR) 
#endif

struct Module {
    unsigned int hitStart;
    unsigned int hitNums;
    float z;

    __device__ Module(){}
    __device__ Module(const unsigned int _hitStart, const unsigned int _hitNums, const float _z) : 
        hitStart(_hitStart), hitNums(_hitNums), z(_z) {}
};

struct Hit {
    float x;
    float y;

    __device__ Hit(){}
    __device__ Hit(const float _x, const float _y) :
        x(_x), y(_y) {}
};

struct Track { // 4 + 24 * 4 = 100 B
    unsigned int hitsNum;
    unsigned int hits[MAX_TRACK_SIZE];

    __device__ Track(){}
    __device__ Track(const unsigned int _hitsNum, const unsigned int _h0, const unsigned int _h1, const unsigned int _h2) : 
        hitsNum(_hitsNum) {
        
        hits[0] = _h0;
        hits[1] = _h1;
        hits[2] = _h2;
    }
};
