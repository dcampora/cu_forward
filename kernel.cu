
#include "kernel.cuh"

__global__ void parallelSearch(track *tracks, char *input, int *num_tracks)
{
	// Prepare input
	if(threadIdx.x == 0){
		num_tracks = 0;

		int dataOffset = max_hits * num_events;
		
		numHitsEvents = (int*) &input[0];
		sensor_hitStarts = (int*) (numHitsEvents + num_events);
		sensor_hitsNums = (int*) (sensor_hitStarts + num_events * sens_num);
		sensor_Zs = (float*) (sensor_hitsNums + num_events * sens_num);
	
		hit_ids = (int*) (sensor_Zs + num_events * sens_num);
		track_ids = hit_ids +  dataOffset;
		hit_sensorNums = track_ids + dataOffset;
		hit_isUseds = hit_sensorNums + dataOffset;
		hit_Xs = (float*) (hit_isUseds + dataOffset);
		hit_Ys = hit_Xs + dataOffset;
		hit_Zs = hit_Ys + dataOffset;
		hit_Ws = hit_Zs + dataOffset;
	}
	__syncthreads();

	__shared__ track shared_fracks;

	// Now we're talking! :)
	int event_no = blockIdx.x;

	int event_sensor_displ = event_no * sens_num;
	int event_hit_displ = event_no * max_hits;

	// sensorInfo sensor0, sensor1, sensor2;
	int sens0, sens1, sens2;

	// For the moment only odd
	for(sens0 = sens_num - 1; sens0 >= 2; sens0 = sens0 - 2){
		sens1 = sens0 - 2;

		// Maybe a more intelligent way to do this?
		int sensor0_startPosition = sensor_hitStarts[event_sensor_displ + sens0];
		int sensor0_hitsNum = sensor_hitsNums[event_sensor_displ + sens0];
		int sensor0_z = sensor_Zs[event_sensor_displ + sens0];
		int sensor1_startPosition = sensor_hitStarts[event_sensor_displ + sens1];
		int sensor1_hitsNum	= sensor_hitsNums[event_sensor_displ + sens1];
		int sensor1_z = sensor_Zs[event_sensor_displ + sens1];
		
		int dxMax = f_m_maxXSlope * fabs( sensor1_z - sensor0_z );
		int dyMax = f_m_maxYSlope * fabs( sensor1_z - sensor0_z );

		// Process current tracks
		
	}


}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t invokeParallelSearch(int numBlocks, int numThreads,
	char* input, int size, track*& tracks, int& num_tracks)
{
    char *dev_input = 0;
	track *dev_tracks = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for max_tracks tracks
    cudaStatus = cudaMalloc((void**)&dev_tracks, max_tracks * 66);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 1 failed!");
        goto Error;
    }

    // Copy input file from host memory to GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_input, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 2 failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    parallelSearch<<<numBlocks, numThreads>>>(dev_tracks, dev_input, &num_tracks);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(tracks, dev_tracks, num_tracks * 66, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_tracks);
	cudaFree(dev_input);
    
    return cudaStatus;
}