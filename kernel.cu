
#include "kernel.cuh"

__global__ void prepareData(char* input, int* _prevs, int* _nexts){
	no_sensors = (int*) &input[0];
    no_hits = (int*) (no_sensors + 1);
    sensor_Zs = (int*) (no_hits + 1);
    sensor_hitStarts = (int*) (sensor_Zs + no_sensors[0]);
    sensor_hitNums = (int*) (sensor_hitStarts + no_sensors[0]);
    hit_IDs = (int*) (sensor_hitNums + no_sensors[0]);
    hit_Xs = (double*) (hit_IDs + no_hits[0]);
	hit_Ys = (double*) (hit_Xs + no_hits[0]);
	hit_Zs = (int*) (hit_Ys + no_hits[0]);

	prevs = _prevs;
	nexts = _nexts;
}

__global__ void neighboursFinder()
{
	__shared__ Hit prev_hits[HITS_SHARED];
	__shared__ Hit next_hits[HITS_SHARED];
	__shared__ Sensor s[3];

	/*
	gridDim.{x,y,z}
	blockIdx.{x,y,z}

	blockDim.{x,y,z}
	threadIdx.{x,y,z}
	*/

	int current_sensor, prev_sensor, next_sensor, address;

	current_sensor = blockIdx.x;
	prev_sensor = current_sensor - 2;
	next_sensor = current_sensor + 2;

	// Prepare input
	if (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2){
		
		// trick to execute things in the same warp
		bool condition = (prev_sensor >= 0 && threadIdx.x == 0) || (next_sensor <= 48 && threadIdx.x == 2) || threadIdx.x == 1;
		if (condition){
			address = prev_sensor * (threadIdx.x==0) +
				current_sensor * (threadIdx.x==1) + next_sensor * (threadIdx.x==2);
			s[threadIdx.x].z = sensor_Zs[address];
			s[threadIdx.x].hitStart = sensor_hitStarts[address];
			s[threadIdx.x].hitNums = sensor_hitNums[address];
		}
	}

	__syncthreads();

	// TODO: Account for special cases (2 first sensors, and 2 last sensors)
	if (prev_sensor >= 0 && next_sensor <= 48){

		int prev_num_hits_to_load = int(ceilf(s[0].hitNums / blockDim.x));
		int current_num_hits_to_load = int(ceilf(s[1].hitNums / blockDim.x));
		int next_num_hits_to_load = int(ceilf(s[2].hitNums / blockDim.x));
		Hit current_hit;
		
		float best_fit = MAX_FLOAT;
		int best_prev = -1;
		int best_next = -1;

		// Load elements into
		// - current_hit: The element we are treating
		// - prev_hits:   Previous hits (HITS_SHARED === blockDim.x)
		// - next_hits:   Next hits (HITS_SHARED === blockDim.x)

		for (int i=0; i<current_num_hits_to_load; ++i){
			int current_element = s[1].hitStart + i * blockDim.x + threadIdx.x;

			if (current_element < s[1].hitNums){
				current_hit.x = hit_Xs[current_element];
				current_hit.y = hit_Ys[current_element];
			}

			for (int j=0; j<prev_num_hits_to_load; ++j){
				int prev_element = s[0].hitStart + j * blockDim.x + threadIdx.x;

				if (prev_element < s[0].hitNums){
					prev_hits[threadIdx.x].x = hit_Xs[prev_element];
					prev_hits[threadIdx.x].y = hit_Ys[prev_element];
				}

				for (int k=0; k<next_num_hits_to_load; ++k){
					int next_element = s[2].hitStart + k * blockDim.x + threadIdx.x;

					if (next_element < s[2].hitNums){
						next_hits[threadIdx.x].x = hit_Xs[next_element];
						next_hits[threadIdx.x].y = hit_Ys[next_element];
					}

					// Start comparison, minimize the best_fit for each current_element
					if(current_element < s[1].hitNums){
						// Minimize best fit
						for (int m=0; m<HITS_SHARED; ++m){
							for (int n=0; n<HITS_SHARED; ++n){
								float fit;
								// fit = prev_hits[m]

								float t = s[1].z - s[0].z / s[2].z - s[0].z;
								float x = prev_hits[m].x + t * (next_hits[n].x - prev_hits[m].x);
								float y = prev_hits[m].y + t * (next_hits[n].y - prev_hits[m].y);
								float d1 = sqrtf( powf( (float) (current_hit.x - x), 2.0) + 
												  powf( (float) (current_hit.y - y), 2.0));

								t = - s[0].z / s[2].z - s[0].z;
								x = prev_hits[m].x + t * (next_hits[n].x - prev_hits[m].x);
								y = prev_hits[m].y + t * (next_hits[n].y - prev_hits[m].y);
								float d2 = sqrtf( powf( (float) (current_hit.x - x), 2.0) + 
												  powf( (float) (current_hit.y - y), 2.0));

								fit = powf(d1, 2.0) + d2;
								
								bool fit_is_better = fit < best_fit;
								best_fit = fit_is_better * fit + !fit_is_better * best_fit;
								best_prev = fit_is_better * (s[0].hitStart + j * blockDim.x + m) +
											!fit_is_better * best_prev;
								best_next = fit_is_better * (s[2].hitStart + k * blockDim.x + n) + 
											!fit_is_better * best_next;
							}
						}
					}
				}
			}

			// Store best fit into solution array.
			if(current_element < s[1].hitNums){
				prevs[current_element] = best_prev;
				nexts[current_element] = best_next;
			}
		}
	}
}

