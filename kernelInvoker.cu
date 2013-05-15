
#include "kernelInvoker.cuh"
#include "kernel.cuh"

#define cudaCheck(stmt) do {										\
        cudaError_t err = stmt;										\
        if (err != cudaSuccess) {									\
            std::cerr << "Failed to run " << #stmt << std::endl;    \
            return err;										        \
        }															\
    } while(0)

// Helper function for using CUDA to add vectors in parallel.
cudaError_t invokeParallelSearch(int numBlocks, int numThreads,
	char* input, int size, Track*& tracks, int*& num_tracks){
    
	char *dev_input = 0;
	int* dev_num_tracks = 0;
	Track *dev_tracks = 0;
    cudaError_t cudaStatus = cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaCheck(cudaSetDevice(0));
    
	// Allocate memory
	// Allocate CPU buffers
	tracks = (Track*) malloc(max_tracks * sizeof(Track));
	num_tracks = (int*) malloc(sizeof(int));

    // Allocate GPU buffers
    cudaCheck(cudaMalloc((void**)&dev_tracks, max_tracks * sizeof(Track)));
    
    // Copy input file from host memory to GPU buffers
    cudaCheck(cudaMalloc((void**)&dev_input, size));
    cudaCheck(cudaMalloc((void**)&dev_num_tracks, sizeof(int)));
    
	// memcpys
    cudaCheck(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
    
    // Launch a kernel on the GPU with one thread for each element.
    parallelSearch<<<numBlocks, numThreads>>>(dev_tracks, dev_input, dev_num_tracks);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaCheck(cudaDeviceSynchronize());
    
	// cuda copy back
	cudaCheck(cudaMemcpy(num_tracks, dev_num_tracks, sizeof(int), cudaMemcpyDeviceToHost));
	
    // Copy output vector from GPU buffer to host memory.
    cudaCheck(cudaMemcpy(tracks, dev_tracks, num_tracks[0] * sizeof(Track), cudaMemcpyDeviceToHost));
    
    return cudaStatus;
}
