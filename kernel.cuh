
#include "Definitions.cuh"

extern int max_tracks;

__global__ void parallelSearch(track *tracks, char *input, int &num_tracks);

cudaError_t invokeParallelSearch(int numBlocks, int numThreads,
	char* input, int size, track*& tracks, int& num_tracks);
