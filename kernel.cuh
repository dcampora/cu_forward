
#include "Definitions.cuh"

#define HITS_SHARED 32
#define MAX_FLOAT 100000000.0

struct Sensor {
	int z;
	int hitStart;
	int hitNums;
};

struct Hit {
	float x;
	float y;
};

__global__ void prepareData(char* input, int* _prevs, int* _nexts);
__global__ void neighboursFinder();
