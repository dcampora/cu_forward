
#include "Definitions.cuh"
#include "kernel.cuh"

#include "Tools.h"
#include "kernelInvoker.cuh"

int max_tracks = 1000;

int main()
{
	// Read file (s)
	char* input;
	int size;
	std::string c = "pixel-sft-event-0.dump";
	readFile(c.c_str(), input, size);

	// Return elements
	Track* tracks;
	int* num_tracks;

	int numBlocks = 1, numThreads = 1;

	// Pre-processing, quick sort over X
	quickSortInput(input);

    cudaError_t cudaStatus = invokeParallelSearch(numBlocks, numThreads, input, size, tracks, num_tracks);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cuda kernel failed" << std::endl;
        return cudaStatus;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed" << std::endl;
        return cudaStatus;
    }

	std::cout << "Everything went quite well!" << std::endl;

	getchar();

    return 0;
}
