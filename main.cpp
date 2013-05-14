
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cstdio>

#include "kernel.cuh"
using namespace std;

void readFile(string filename, char*& input, int& size);

// Variables set up by hand
int max_tracks = 400;

int main()
{
	track* tracks;
	int num_tracks;

	// Read file
	char* input;
	int size;
	string c = "input_float.dump";
	readFile(c.c_str(), input, size);

	int numBlocks = 1, numThreads = 1;

    cudaError_t cudaStatus = invokeParallelSearch(numBlocks, numThreads, input, size, tracks, num_tracks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda kernel failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	getchar();

    return 0;
}

void readFile(string filename, char*& input, int& size){
	// Give me them datas!!11!
	ifstream infile (filename.c_str(), ifstream::binary);

	// get size of file
	infile.seekg(0, ifstream::end);
	size = infile.tellg();
	infile.seekg(0);

	// read content of infile with pointers
	input = (char*) malloc(size);
	infile.read (input, size);
	infile.close();
}
