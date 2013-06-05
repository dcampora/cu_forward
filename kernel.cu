
#include "kernel.cuh"

__global__ void prepareData(char* input, int* _prevs, int* _nexts){
	no_sensors = (int*) &input[0];
    no_hits = (int*) (no_sensors + 1);
    sensor_Zs = (int*) (no_hits + 1);
    sensor_hitStarts = (int*) (sensor_Zs + no_sensors[0]);
    sensor_hitNums = (int*) (sensor_hitStarts + no_sensors[0]);
    hit_IDs = (int*) (sensor_hitNums + no_sensors[0]);
    hit_Xs = (float*) (hit_IDs + no_hits[0]);
	hit_Ys = (float*) (hit_Xs + no_hits[0]);
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

	Hit current_hit;
	float best_fit;
	int best_prev;
	int best_next;
	int current_element;
	int next_num_hits_to_load;
	int prev_num_hits_to_load;
	int current_num_hits_to_load;
	int prev_element;
	int next_element;
	float fit, t, x, y, d1, d2;
	bool fit_is_better;

	// TODO: Account for special cases (2 first sensors, and 2 last sensors)
	// if (prev_sensor < 0 || next_sensor > NUM_SENSORS){
	if(false){
		if (next_sensor > NUM_SENSORS){
			current_sensor -= 2;
			next_sensor -= 2;
		}

		current_num_hits_to_load = int(ceilf(s[1].hitNums / blockDim.x));
		next_num_hits_to_load = int(ceilf(s[2].hitNums / blockDim.x));
		
		best_fit = MAX_FLOAT;
		best_prev = -1;
		best_next = -1;
		
		// Load elements into
		// - current_hit: The element we are treating
		// - prev_hits:   Previous hits (HITS_SHARED === blockDim.x)
		// - next_hits:   Next hits (HITS_SHARED === blockDim.x)

		for (int i=0; i<current_num_hits_to_load; ++i){
			current_element = i * blockDim.x + threadIdx.x;

			if (current_element < s[1].hitNums){
				current_hit.x = hit_Xs[s[1].hitStart + current_element];
				current_hit.y = hit_Ys[s[1].hitStart + current_element];
			}

			for (int k=0; k<next_num_hits_to_load; ++k){
				next_element = k * blockDim.x + threadIdx.x;

				if (next_element < s[2].hitNums){
					next_hits[threadIdx.x].x = hit_Xs[s[2].hitStart + next_element];
					next_hits[threadIdx.x].y = hit_Ys[s[2].hitStart + next_element];
				}

				// Start comparison, minimize the best_fit for each current_element
				if(current_element < s[1].hitNums){
					// Minimize best fit
					for (int m=0; m<HITS_SHARED; ++m){
						// float fit;
						// fit = prev_hits[m]

						/* Special cases calculation
						d is h0-h1 distance to <0,0,0> on plane s0.
						*/

						t = - s[1].z / s[2].z - s[1].z;
						x = current_hit.x + t * (next_hits[m].x - current_hit.x);
						y = current_hit.y + t * (next_hits[m].y - current_hit.y);
						fit = powf( (float) (x), 2.0) + 
							  powf( (float) (y), 2.0);
								
						fit_is_better = fit < best_fit;
						best_fit = fit_is_better * fit + !fit_is_better * best_fit;
						best_next = fit_is_better * (s[2].hitStart + k * blockDim.x + m) + 
									!fit_is_better * best_next;
					}
				}
			}
		}
		
		// Store best fit into solution array.
		if(prev_sensor < 0){
			nexts[s[1].hitStart + current_element] = best_next;
		}
		else {
			prevs[s[1].hitStart + current_element] = best_next;
		}
	}

	if (prev_sensor >= 0 && next_sensor <= NUM_SENSORS){

		prev_num_hits_to_load = int(ceilf(s[0].hitNums / blockDim.x));
		current_num_hits_to_load = int(ceilf(s[1].hitNums / blockDim.x));
		next_num_hits_to_load = int(ceilf(s[2].hitNums / blockDim.x));
		
		best_fit = MAX_FLOAT;
		best_prev = -1;
		best_next = -1;

		// Load elements into
		// - current_hit: The element we are treating
		// - prev_hits:   Previous hits (HITS_SHARED === blockDim.x)
		// - next_hits:   Next hits (HITS_SHARED === blockDim.x)

		for (int i=0; i<current_num_hits_to_load; ++i){
			current_element = i * blockDim.x + threadIdx.x;

			if (current_element < s[1].hitNums){
				current_hit.x = hit_Xs[s[1].hitStart + current_element];
				current_hit.y = hit_Ys[s[1].hitStart + current_element];
			}

			for (int j=0; j<prev_num_hits_to_load; ++j){
				prev_element = j * blockDim.x + threadIdx.x;

				if (prev_element < s[0].hitNums){
					prev_hits[threadIdx.x].x = hit_Xs[s[0].hitStart + prev_element];
					prev_hits[threadIdx.x].y = hit_Ys[s[0].hitStart + prev_element];
				}

				for (int k=0; k<next_num_hits_to_load; ++k){
					next_element = k * blockDim.x + threadIdx.x;

					if (next_element < s[2].hitNums){
						next_hits[threadIdx.x].x = hit_Xs[s[2].hitStart + next_element];
						next_hits[threadIdx.x].y = hit_Ys[s[2].hitStart + next_element];
					}

					// Start comparison, minimize the best_fit for each current_element
					if(current_element < s[1].hitNums){
						// Minimize best fit
						for (int m=0; m<HITS_SHARED; ++m){
							for (int n=0; n<HITS_SHARED; ++n){
								// float fit;
								// fit = prev_hits[m]

								/* Calculation of the best fit:
								hits on sensors 0, 1 and 2 are h0, h1 and h2. We are calculating
								the best h0 and h2 for h1.

								d1 is the distance from the line h0-h2 to h1 in plane sensor s1.
								d2 is the distance from the line h0-h2 to <0,0,0> in plane sensor s0.
								*/

								t = s[1].z - s[0].z / s[2].z - s[0].z;
								x = prev_hits[m].x + t * (next_hits[n].x - prev_hits[m].x);
								y = prev_hits[m].y + t * (next_hits[n].y - prev_hits[m].y);
								d1 = sqrtf( powf( (float) (current_hit.x - x), 2.0) + 
											powf( (float) (current_hit.y - y), 2.0));

								t = - s[0].z / s[2].z - s[0].z;
								x = prev_hits[m].x + t * (next_hits[n].x - prev_hits[m].x);
								y = prev_hits[m].y + t * (next_hits[n].y - prev_hits[m].y);
								d2 = sqrtf( powf( (float) (x), 2.0) + 
											powf( (float) (y), 2.0));

								// fit = powf(d1, 2.0) + d2;
								fit = d1;
								
								fit_is_better = fit < best_fit;
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
				prevs[s[1].hitStart + current_element] = best_prev;
				nexts[s[1].hitStart + current_element] = best_next;
			}
		}
	}
}

__global__ void neighboursCleaner()
{
	int block_size = int(ceilf( ((float)no_hits[0]) / gridDim.x));
	int thread_size = int(ceilf( ((float)block_size) / blockDim.x));

	for(int j=0; j<thread_size; ++j){
		int current_hit = blockIdx.x * block_size + blockDim.x * j + threadIdx.x;
		if(current_hit < no_hits[0]){
			int next_hit = nexts[current_hit];
			if(next_hit < 0 || prevs[next_hit] != current_hit){
				prevs[next_hit] = -1;
				nexts[current_hit] = -1;
			}
		}
	}
}

/** fitHits, gives the fit between h0 and h1.

The accept condition requires dxmax and dymax to be in a range.

The fit (d1) depends on the distance of the tracklet to <0,0,0>.
*/
__device__ float fitHits(Hit& h0, Hit& h1, Sensor& s0, Sensor& s1){
	// Max dx, dy permissible over next hit

	// TODO: This can go outside this function (only calc once per pair
	// of sensors). Also, it could only be calculated on best fitting distance d1.
	float s_dist = fabs((float)( s1.z - s0.z ));
	float dxmax = PARAM_MAXXSLOPE * s_dist;
	float dymax = PARAM_MAXYSLOPE * s_dist;
	
	bool accept_condition = fabs(h1.x - h0.x) < dxmax &&
							fabs(h1.y - h0.y) < dymax;

	/*float dxmax = PARAM_MAXXSLOPE * fabs((float)( s1.z - s0.z ));
	float dymax = PARAM_MAXYSLOPE * fabs((float)( s1.z - s0.z ));*/
	
	// Distance to <0,0,0> in its XY plane.
	float t = - s0.z / (s1.z - s0.z);
	float x = h0.x + t * (h1.x - h0.x);
	float y = h0.y + t * (h1.y - h0.y);
	float d1 = sqrtf( powf( (float) (x), 2.0) + 
				powf( (float) (y), 2.0));

	return accept_condition * d1 + !accept_condition * MAX_FLOAT;
}

// TODO: Optimize with Olivier's
__device__ float fitHitToTrack(Track& t, Hit& h1, Sensor& s1){
	// tolerance
	// TODO: To improve efficiency, try with PARAM_TOLERANCE_EXTENDED
	float x_prediction = t.x0 + t.tx * s1.z;
	bool tol_condition = fabs(x_prediction - h1.x) < PARAM_TOLERANCE;

	// chi2
	float dx = x_prediction - h1.x;
	float dy = (t.y0 + t.ty * s1.z) - h1.y;
	float chi2 = dx * dx * PARAM_W + dy * dy * PARAM_W;

	// TODO: The check for chi2_condition can totally be done after this call
	bool chi2_condition = chi2 < PARAM_MAXCHI2;
	
	return tol_condition * chi2_condition * chi2 + (!tol_condition || !chi2_condition) * MAX_FLOAT;
}

// Create track
__device__ void acceptTrack(Track& t, Hit& h0, Hit& h1, Sensor& s0, Sensor& s1, int h0_num, int h1_num){
	float wz = PARAM_W * s0.z;

	t.s0 = PARAM_W;
	t.sx = PARAM_W * h0.x;
	t.sz = wz;
	t.sxz = wz * h0.x;
	t.sz2 = wz * s0.z;

	t.u0 = PARAM_W;
	t.uy = PARAM_W * h0.y;
	t.uz = wz;
	t.uyz = wz * h0.y;
	t.uz2 = wz * s0.z;

	t.hitsNum = 1;
	t.hits[0] = h0_num;

	// note: This could be done here (inlined)
	updateTrack(t, h1, s1, h1_num);
}

// Update track
__device__ void updateTrack(Track& t, Hit& h1, Sensor& s1, int h1_num){
	float wz = PARAM_W * s1.z;

	t.s0 += PARAM_W;
	t.sx += PARAM_W * h1.x;
	t.sz += wz;
	t.sxz += wz * h1.x;
	t.sz2 += wz * s1.z;

	t.u0 += PARAM_W;
	t.uy += PARAM_W * h1.y;
	t.uz += wz;
	t.uyz += wz * h1.y;
	t.uz2 += wz * s1.z;

	t.hits[t.hitsNum] = h1_num;
	t.hitsNum++;

	updateTrackCoords(t);
}

// TODO: Check this function
__device__ void updateTrackCoords (Track& t){
	float den = ( t.sz2 * t.s0 - t.sz * t.sz );
	if ( fabs(den) < 10e-10 ) den = 1.f;
	t.tx     = ( t.sxz * t.s0  - t.sx  * t.sz ) / den;
	t.x0     = ( t.sx  * t.sz2 - t.sxz * t.sz ) / den;

	den = ( t.uz2 * t.u0 - t.uz * t.uz );
	if ( fabs(den) < 10e-10 ) den = 1.f;
	t.ty     = ( t.uyz * t.u0  - t.uy  * t.uz ) / den;
	t.y0     = ( t.uy  * t.uz2 - t.uyz * t.uz ) / den;
}

/** Simple implementation of the Kalman Filter selection on the GPU (step 4).

Will rely on pre-processing for selecting next-hits for each hit.

Implementation,
- Perform implementation searching on all hits for each sensor

The algorithm has two parts:
- Track creation (two hits)
- Track following (consecutive sensors)


Optimizations,
- Optimize with shared memory
- Optimize further with pre-processing

Then there must be a post-processing, which selects the
best tracks based on (as per the conversation with David):
- length
- chi2

For this, simply use the table with all created tracks:

TODO:
#track, h0, h1, h2, h3, ..., hn, length, chi2

*/

__global__ void gpuKalman(Track* tracks, bool* track_holders){
	Track t;
	Sensor s0, s1;
	Hit h0, h1;

	float fit, best_fit;
	bool fit_is_better, accept_track;
	int best_hit, current_hit;

	int current_sensor = (47 - blockIdx.x);

	s0.hitStart = sensor_hitStarts[current_sensor];
	s0.hitNums = sensor_hitNums[current_sensor];
	s0.z = sensor_Zs[current_sensor];

	// Analyze the best hit for next sensor
	int next_sensor = current_sensor - 2;
	
	// TODO: Delete these infamous lines
	for(int i=0; i<int(ceilf(s0.hitNums / blockDim.x)); ++i){
		current_hit = blockIdx.x * i + threadIdx.x;
		if(current_hit < s0.hitNums){
			track_holders[s0.hitStart + current_hit] = false;
		}
	}

	if(next_sensor >= 0){
		// Iterate in all hits for current sensor
		for(int i=0; i<int(ceilf(s0.hitNums / blockDim.x)); ++i){
			current_hit = blockIdx.x * i + threadIdx.x;

			h0.x = hit_Xs[ s0.hitStart + current_hit ];
			h0.y = hit_Ys[ s0.hitStart + current_hit ];
			// t.x0 = h0.x;
			// t.y0 = h0.y;
			
			// Initialize track
			for(int j=0; j<TRACK_SIZE; ++j){
				t.hits[j] = -1;
			}

			if(current_hit < s0.hitNums){
				// TODO: shared memory.
				s1.hitStart = sensor_hitStarts[next_sensor];
				s1.hitNums = sensor_hitNums[next_sensor];
				s1.z = sensor_Zs[next_sensor];
		
				// TRACK CREATION
				// TODO: Modify with preprocessed list of hits.
				best_fit = MAX_FLOAT;
				best_hit = -1;
				for(int j=0; j<sensor_hitNums[next_sensor]; ++j){
					// TODO: Load in chunks of SHARED_MEMORY and take
					// them from shared memory.
					h1.x = hit_Xs[s1.hitStart + j];
					h1.y = hit_Ys[s1.hitStart + j];

					fit = fitHits(h0, h1, s0, s1);
					fit_is_better = fit < best_fit;

					best_fit = fit_is_better * fit + !fit_is_better * best_fit;
					best_hit = fit_is_better * j + !fit_is_better * best_hit;
				}

				accept_track = best_fit != MAX_FLOAT;

				// We have a best fit!

				// For those who have tracks, we go on
				if(accept_track){
					// Fill in t (ONLY in case the best fit is acceptable)
					acceptTrack(t, h0, h1, s0, s1, s0.hitStart + current_hit, s1.hitStart + best_hit);

					// TRACK FOLLOWING
					next_sensor -= 2;
					while(next_sensor >= 0){
						// Go to following sensor
						/*s0.hitNums = s1.hitNums;
						s0.hitStart = s1.hitStart;
						s0.z = s1.z;*/
						
						s1.hitStart = sensor_hitStarts[next_sensor];
						s1.hitNums = sensor_hitNums[next_sensor];
						s1.z = sensor_Zs[next_sensor];

						best_fit = MAX_FLOAT;
						for(int k=0; k<sensor_hitNums[next_sensor]; ++k){
							// TODO: Load in chunks of SHARED_MEMORY and take
							// them from shared memory.
							h1.x = hit_Xs[s1.hitStart + k];
							h1.y = hit_Ys[s1.hitStart + k];

							fit = fitHitToTrack(t, h1, s1);
							fit_is_better = fit < best_fit;

							best_fit = fit_is_better * fit + !fit_is_better * best_fit;
							best_hit = fit_is_better * k + !fit_is_better * best_hit;
						}

						// We have a best fit!
						// Fill in t, ONLY in case the best fit is acceptable

						// TODO: Maybe try to do this more "parallel"
						if(best_fit != MAX_FLOAT){
							updateTrack(t, h1, s1, s1.hitStart + best_hit);
						}

						next_sensor -= 2;
					}
				}

				// If it's a track, write it to memory, no matter what kind
				// of track it is.
				// TODO: Weird problem
				track_holders[s0.hitStart + current_hit] = accept_track && (t.hitsNum > 2);
				if(accept_track && (t.hitsNum > 2)){
					tracks[s0.hitStart + current_hit] = t;
				}
			}
		}
	}
}
