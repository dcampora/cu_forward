
#include "Definitions.cuh"
#include "kernel.cuh"

#include "Tools.h"
#include "kernelInvoker.cuh"

// TODO: debug purposes only
extern float* h_hit_Xs;
extern float* h_hit_Ys;
extern int*   h_hit_Zs;

extern int*   h_prevs;
extern int*   h_nexts;

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

	dim3 numBlocks(48), numThreads(32);

	// Pre-processing, quick sort over X
	// quickSortInput(input);

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

	/*
	VELOModel v;
	osgViewer::Viewer viewer;
	initializeModel(std::string("pixel-sft-event-0.dump"), v, viewer);
	// Add tracks to the model
	for(int i=0; i<h_no_hits[0]; ++i){
		int phit = h_prevs[i];
		if(phit >= 0)
			v.addTrack(osg::Vec3f(h_hit_Xs[phit], h_hit_Ys[phit], (float) h_hit_Zs[phit]),
					   osg::Vec3f(h_hit_Xs[i], h_hit_Ys[i], (float) h_hit_Zs[i]));
	}
	while(1)
		viewer.run();
	*/

	getchar();

    return 0;
}
