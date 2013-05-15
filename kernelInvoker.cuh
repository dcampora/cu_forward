
#include <iostream>
#include "Definitions.cuh"

extern int max_tracks;

cudaError_t invokeParallelSearch(int numBlocks, int numThreads,
	char* input, int size, Track*& tracks, int*& num_tracks);

