
#include <iostream>
#include "Definitions.cuh"

extern int* h_no_hits;

void getMaxNumberOfHits(char*& input, int& maxHits);
cudaError_t invokeParallelSearch(dim3 numBlocks, dim3 numThreads,
	char* input, int size, Track*& tracks, int*& num_tracks);

