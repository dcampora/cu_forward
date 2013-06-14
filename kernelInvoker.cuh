
#include <iostream>
#include "Definitions.cuh"
#include "Histo.h"

#include <fstream>

extern int* h_no_hits;

void getMaxNumberOfHits(char*& input, int& maxHits);
cudaError_t invokeParallelSearch(dim3 numBlocks, dim3 numThreads,
	char* input, int size, Track*& tracks, int*& num_tracks);

void printTrack(Track* tracks, int i);
void printOutSensorHits(int sensorNumber, int* prevs, int* nexts);
void printOutAllSensorHits(int* prevs, int* nexts);
