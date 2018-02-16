#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

// Number of threads
#define NUMTHREADS_X 32

// How many concurrent h1s to process max
// It should be a divisor of NUMTHREADS_X
#define MAX_CONCURRENT_H1 16

// Number of concurrent h1s in the first iteration
// The first iteration has no flagged hits and more triplets per hit
#define MAX_CONCURRENT_H1_FIRST_ITERATION 8

// These parameters impact the found tracks
// Maximum / minimum acceptable phi
// This impacts enourmously the speed of track seeding
#define PHI_EXTRAPOLATION 0.062

// Tolerance angle for forming triplets
#define MAX_SLOPE 0.4f
#define TOLERANCE 0.6f

// Maximum scatter of each three hits
// This impacts velo tracks and a to a lesser extent
// long and long strange tracks
#define MAX_SCATTER_SEEDING 0.004f

// Making a bigger forwarding scatter window causes
// less clones and more ghosts
#define MAX_SCATTER_FORWARDING 0.004f

// Number of seeding iterations before storing tracklets
// This impacts the amount of shared memory to request per thread
// #define SEEDING_CONTINUOUS_ITERATIONS 2

// Maximum number of skipped modules allowed for a track
// before storing it
#define MAX_SKIPPED_MODULES 3

// Total number of atomics required
// This is just a constant (that I keep changing)
#define NUM_ATOMICS 4

// Constants for requested storage on device
#define MAX_TRACKS 3000
#define MAX_TRACK_SIZE 24
#define MAX_NUMHITS_IN_MODULE 256

// Maximum number of tracks to follow at a time
#define TTF_MODULO 2000

// Run over same data several times to stress processor
// (ie. increase the runtime of kernels)
#define DO_REPEATED_EXECUTION false
#define REPEAT_ITERATIONS 100

// Parameters to print out solutions
#define PRINT_SOLUTION false
#define PRINT_FILL_CANDIDATES false
#define PRINT_VERBOSE false
#define PRINT_BINARY false
#define ASSERTS_ENABLED true
#define RESULTS_FOLDER "results"

#if ASSERTS_ENABLED == true
#include "assert.h"
#define ASSERT(EXPR) assert(EXPR);
#else
#define ASSERT(EXPR) 
#endif

struct Module {
    unsigned short hitStart;
    unsigned short hitNums;
    float z;

    __device__ Module(){}
    __device__ Module(
      const unsigned short _hitStart,
      const unsigned short _hitNums,
      const float _z
    ) : hitStart(_hitStart), hitNums(_hitNums), z(_z) {}
};

struct Hit {
    float x;
    float y;

    __device__ Hit(){}
    __device__ Hit(
      const float _x,
      const float _y
    ) : x(_x), y(_y) {}
};

struct Track { // 2 + 24 * 2 = 50 B
  unsigned short hitsNum;
  unsigned short hits[MAX_TRACK_SIZE];

  __device__ Track(){}
  __device__ Track(
    const unsigned short _hitsNum,
    const unsigned short _h0,
    const unsigned short _h1,
    const unsigned short _h2
  ) : hitsNum(_hitsNum) {
    hits[0] = _h0;
    hits[1] = _h1;
    hits[2] = _h2;
  }
};
