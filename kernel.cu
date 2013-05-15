
#include "kernel.cuh"

__global__ void parallelSearch(Track *tracks, char *input, int *num_tracks)
{
	// Prepare input
	if(threadIdx.x == 0){
		num_tracks[0] = 0;

		no_sensors = (int*) &input[0];
        no_hits = (int*) (no_sensors + 1);
        sensor_Zs = (int*) (no_hits + 1);
        sensor_hitStarts = (int*) (sensor_Zs + no_sensors[0]);
        sensor_hitNums = (int*) (sensor_hitStarts + no_sensors[0]);
        hit_IDs = (int*) (sensor_hitNums + no_sensors[0]);
        hit_Xs = (double*) (hit_IDs + no_hits[0]);
		hit_Ys = (double*) (hit_Xs + no_hits[0]);
		hit_Zs = (int*) (hit_Ys + no_hits[0]);

        hits_num = no_hits[0];
	}
	__syncthreads();

	if(threadIdx.x == 0){
		num_tracks[0] = 1;
	}
}
