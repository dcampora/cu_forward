
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
cudaError_t invokeParallelSearch(dim3 numBlocks, dim3 numThreads,
	char* input, int size, Track*& tracks, int*& num_tracks){
    
	char *dev_input = 0;
	int* dev_num_tracks = 0;
	Track *dev_tracks = 0;
	int* dev_prevs = 0;
	int* dev_nexts = 0;
    cudaError_t cudaStatus = cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaCheck(cudaSetDevice(0));
    
	// Allocate memory
	// Allocate CPU buffers
	tracks = (Track*) malloc(MAX_TRACKS * sizeof(Track));
	num_tracks = (int*) malloc(sizeof(int));

    // Allocate GPU buffers
    cudaCheck(cudaMalloc((void**)&dev_tracks, MAX_TRACKS * sizeof(Track)));
	cudaCheck(cudaMalloc((void**)&dev_prevs, h_no_hits[0] * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&dev_nexts, h_no_hits[0] * sizeof(int)));
    
    // Copy input file from host memory to GPU buffers
    cudaCheck(cudaMalloc((void**)&dev_input, size));
    cudaCheck(cudaMalloc((void**)&dev_num_tracks, sizeof(int)));
    
	// memcpys
    cudaCheck(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
    
    // Launch a kernel on the GPU with one thread for each element.
	prepareData<<<1, 1>>>(dev_input, dev_prevs, dev_nexts);
    neighboursFinder<<<numBlocks, numThreads>>>();
	neighboursCleaner<<<numBlocks, numThreads>>>();
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaCheck(cudaDeviceSynchronize());
    
	// cuda copy back
	cudaCheck(cudaMemcpy(num_tracks, dev_num_tracks, sizeof(int), cudaMemcpyDeviceToHost));
	
    // Copy output vector from GPU buffer to host memory.
    cudaCheck(cudaMemcpy(tracks, dev_tracks, num_tracks[0] * sizeof(Track), cudaMemcpyDeviceToHost));
    
    return cudaStatus;
}

void getMaxNumberOfHits(char*& input, int& maxHits){
	int* l_no_sensors = (int*) &input[0];
    int* l_no_hits = (int*) (l_no_sensors + 1);
    int* l_sensor_Zs = (int*) (l_no_hits + 1);
    int* l_sensor_hitStarts = (int*) (l_sensor_Zs + l_no_sensors[0]);
    int* l_sensor_hitNums = (int*) (l_sensor_hitStarts + l_no_sensors[0]);

	maxHits = 0;
	for(int i=0; i<l_no_sensors[0]; ++i){
		if(l_sensor_hitNums[i] > maxHits)
			maxHits = l_sensor_hitNums[i];
	}
}