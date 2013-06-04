
#include "Definitions.cuh"

#define HITS_SHARED 32
#define MAX_FLOAT 100000000.0
#define NUM_SENSORS 48

#define PARAM_W 0.0144338f // 0.050 / sqrt( 12. )
#define PARAM_MAXXSLOPE 0.4f
#define PARAM_MAXYSLOPE 0.3f

#define PARAM_TOLERANCE 0.15f
#define PARAM_TOLERANCE_EXTENDED 0.3f


/*
float 	f_m_maxXSlope			= 0.400;
float 	f_m_maxYSlope			= 0.300;
float 	f_m_maxZForRBeamCut		= 200.0;
float 	f_m_maxR2Beam			= 1.0;
int 	f_m_maxMissed			= 4;
float 	f_m_extraTol			= 0.150;
float 	f_m_maxChi2ToAdd		= 100.0;
float 	f_m_maxChi2SameSensor	= 16.0;
float   f_m_maxChi2Short		= 6.0 ;
float   f_m_maxChi2PerHit		= 16.0;
int 	f_m_sensNum				= 48;
float   f_w						= 0.050 / sqrt( 12. );
*/

struct Sensor {
	int z;
	int hitStart;
	int hitNums;
};

struct Hit {
	float x;
	float y;
};

struct Track {
	float x0;
	float y0;
	float tx;
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

	int hitsNum;
	int hits[24];
}


__global__ void prepareData(char* input, int* _prevs, int* _nexts);
__global__ void neighboursFinder();
__global__ void neighboursCleaner();

__global__ void gpuKalman();
