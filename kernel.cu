
#include "kernel.cuh"

__global__ void prepareData(char* input, int* _prevs, int* _nexts, bool* track_holders){
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

	for(int i=0; i<MAX_TRACKS; ++i){
		track_holders[i] = false;
	}
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
__device__ float fitHits(Hit& h0, Hit& h1, Hit &h2, Sensor& s0, Sensor& s1, Sensor& s2){
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
	/*float t = - s0.z / (s1.z - s0.z);
	float x = h0.x + t * (h1.x - h0.x);
	float y = h0.y + t * (h1.y - h0.y);
	float d1 = sqrtf( powf( (float) (x), 2.0f) + 
				powf( (float) (y), 2.0f));*/

	// Distance between the hits.
	// float d1 = sqrtf( powf( (float) (h1.x - h0.x), 2.0f) + 
	//	        powf( (float) (h1.y - h0.y), 2.0f));

	// Distance between line <h0,h1> and h2 in XY plane (s2.z)
	// float t = s2.z - s0.z / (s1.z - s0.z);
	// float x = h0.x + t * (h1.x - h0.x);
	// float y = h0.y + t * (h1.y - h0.y);
	// float d1 = sqrtf( powf( (float) (x - h2.x), 2.0f) + 
	//			powf( (float) (y - h2.y), 2.0f));
	// accept_condition &= (fabs(x - h2.x) < PARAM_TOLERANCE);

	// Require chi2 of third hit below the threshold
	// float t = ((float) (s2.z - s0.z)) / ((float) (s1.z - s0.z));
	float z2_tz = ((float) s2.z - s0.z) / ((float) (s1.z - s0.z));
	float x = h0.x + (h1.x - h0.x) * z2_tz;
	float y = h0.y + (h1.y - h0.y) * z2_tz;

	float dx = x - h2.x;
	float dy = y - h2.y;
	float chi2 = dx * dx * PARAM_W + dy * dy * PARAM_W;
	accept_condition &= chi2 < PARAM_MAXCHI2;

	return accept_condition * chi2 + !accept_condition * MAX_FLOAT;
}

// TODO: Optimize with Olivier's
__device__ float fitHitToTrack(Track& t, Hit& h1, Sensor& s1){
	// tolerance
	// TODO: To improve efficiency, try with PARAM_TOLERANCE_EXTENDED
	float x_prediction = t.x0 + t.tx * s1.z;
	bool tol_condition = fabs(x_prediction - h1.x) < PARAM_TOLERANCE;

	// chi2 of hit (taken out from function for efficiency)
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

For this, simply use the table with all created tracks (postProcess):

#track, h0, h1, h2, h3, ..., hn, length, chi2

*/

__global__ void gpuKalman(Track* tracks, bool* track_holders){
	Track t;
	Sensor s0, s1, s2;
	Hit h0, h1, h2;

	float fit, best_fit;
	bool fit_is_better, accept_track;
	int best_hit, best_hit_h2, current_hit;

	int current_sensor = (47 - blockIdx.x);

	s0.hitStart = sensor_hitStarts[current_sensor];
	s0.hitNums = sensor_hitNums[current_sensor];
	s0.z = sensor_Zs[current_sensor];

	// Analyze the best hit for next sensor
	int next_sensor = current_sensor - 2;
	int third_sensor = current_sensor - 4;
	
	// TODO: Delete these infamous lines
	/* for(int i=0; i<=int(ceilf(s0.hitNums / blockDim.x)); ++i){
		current_hit = blockIdx.x * i + threadIdx.x;
		if(current_hit < s0.hitNums){
			track_holders[s0.hitStart + current_hit] = false;
		}
	} */

	if(third_sensor >= 0){
		// TODO: shared memory.
		s1.hitStart = sensor_hitStarts[next_sensor];
		s1.hitNums = sensor_hitNums[next_sensor];
		s1.z = sensor_Zs[next_sensor];
				
		// TODO: shared memory.
		s2.hitStart = sensor_hitStarts[third_sensor];
		s2.hitNums = sensor_hitNums[third_sensor];
		s2.z = sensor_Zs[third_sensor];

		// Iterate in all hits for current sensor
		for(int i=0; i<int(ceilf( ((float) s0.hitNums) / blockDim.x)); ++i){
			next_sensor = current_sensor - 2;

			current_hit = blockIdx.x * i + threadIdx.x;
			if(current_hit < s0.hitNums){

				h0.x = hit_Xs[ s0.hitStart + current_hit ];
				h0.y = hit_Ys[ s0.hitStart + current_hit ];

				// Initialize track
				for(int j=0; j<TRACK_SIZE; ++j){
					t.hits[j] = -1;
				}
		
				// TRACK CREATION
				// TODO: Modify with preprocessed list of hits.
				best_fit = MAX_FLOAT;
				best_hit = -1;
				best_hit_h2 == -1;
				for(int j=0; j<sensor_hitNums[next_sensor]; ++j){
					// TODO: Load in chunks of SHARED_MEMORY and take
					// them from shared memory.
					h1.x = hit_Xs[s1.hitStart + j];
					h1.y = hit_Ys[s1.hitStart + j];

					// Iterate in the third! list of hits
					for(int k=0; k<sensor_hitNums[third_sensor]; ++k){
						// TODO: Load in chunks of SHARED_MEMORY and take
						// them from shared memory.
						h2.x = hit_Xs[s2.hitStart + k];
						h2.y = hit_Ys[s2.hitStart + k];

						fit = fitHits(h0, h1, h2, s0, s1, s2);
						fit_is_better = fit < best_fit;

						best_fit = fit_is_better * fit + !fit_is_better * best_fit;
						best_hit = fit_is_better * j + !fit_is_better * best_hit;
						best_hit_h2 = fit_is_better * k + !fit_is_better * best_hit_h2;
					}
				}

				accept_track = best_fit != MAX_FLOAT;

				// We have a best fit!

				// For those who have tracks, we go on
				if(accept_track){
					// Fill in t (ONLY in case the best fit is acceptable)
					acceptTrack(t, h0, h1, s0, s1, s0.hitStart + current_hit, s1.hitStart + best_hit);
					updateTrack(t, h2, s2, s2.hitStart + best_hit_h2);

					// TRACK FOLLOWING
					next_sensor -= 4;
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
				track_holders[s0.hitStart + current_hit] = accept_track && (t.hitsNum >= MIN_HITS_TRACK);
				if(accept_track && (t.hitsNum >= MIN_HITS_TRACK)){
					tracks[s0.hitStart + current_hit] = t;
				}
			}
		}
	}
}


/* Calculating the chi2 of a track is quite cumbersome.
It implies loading hit_Xs, hit_Ys, and sensor_Zs elements for each
hit of the track. This introduces branching, and is slow.

However, the track chi2 has to be calculated only when the
track has been created (the tx, ty values change).
*/

__device__ float trackChi2(Track& t){
	float ch = 0.0;
	int nDoF  = -4 + 2 * t.hitsNum;
	Hit h;
	for (int i=0; i<TRACK_SIZE; i++){
		// TODO: Maybe there's a better way to do this
		if(t.hits[i] != -1){
			h.x = hit_Xs[ t.hits[i] ];
			h.y = hit_Ys[ t.hits[i] ];

			ch += hitChi2(t, h, hit_Zs[ t.hits[i] ]);
		}
	}
	return ch/nDoF;
}

__device__ float hitChi2(Track& t, Hit& h, int hit_z){
	// chi2 of a hit
	float dx = (t.x0 + t.tx * hit_z) - h.x;
	float dy = (t.y0 + t.ty * hit_z) - h.y;
	return dx * dx * PARAM_W + dy * dy * PARAM_W;
}


/** The postProcess method takes care of discarding tracks
which are redundant. In other words, it will (hopefully) increase
the purity of our tracks.

- Inspect track_holders and generate track_indexes and num_tracks

The main idea is to accept tracks which have unique (> REQUIRED_UNIQUES) hits.
For this, each track is checked against all other more preferent tracks, and
non common hits are kept.

TODO: Change preference system by something more civilized.
A track t0 has preference over another t1 one if:
t0.hitsNum > t1.hitsNum ||
(t0.hitsNum == t1.hitsNum && chi2(t0) < chi2(t1))
*/
__global__ void postProcess(Track* tracks, bool* track_holders, int* track_indexes, int* num_tracks, int* tracks_to_process){
	// tracks_to_process holds the list of tracks with track_holders[t] == true
	
	// TODO: Try with sh_tracks_to_process
	// __shared__ int sh_tracks_to_process[MAX_POST_TRACKS];

	__shared__ Track sh_tracks[BUNCH_POST_TRACKS];
	__shared__ float sh_chi2[BUNCH_POST_TRACKS];

	__shared__ Track sh_next_tracks[BUNCH_POST_TRACKS];
	__shared__ float sh_next_chi2[BUNCH_POST_TRACKS];
	
	// We will use an atomic to write on a vector concurrently on several values
	__shared__ int tracks_to_process_size;
	__shared__ int tracks_accepted_size;

	tracks_to_process_size = 0;
	tracks_accepted_size = 0;

	__syncthreads(); // for the atomics tracks_to_process_size, and tracks_processed

	int i, j, current_track, next_track;
	bool preferent;

	for(i=0; i<int(ceilf( ((float) no_hits[0]) / blockDim.x)); ++i){
		current_track = blockDim.x * i + threadIdx.x;
		if(current_track < no_hits[0]){
			// Iterate in all tracks (current_track)

			if(track_holders[current_track]){
				// Atomic add
				int current_atomic = atomicAdd(&tracks_to_process_size, 1);

				// TODO: This condition shouldn't exist,
				// redo using method to process in batches if necessary
				// if(current_atomic < MAX_POST_TRACKS)
				tracks_to_process[current_atomic] = current_track;
			}
		}
	}

	__syncthreads();

	// Iterate in all current_tracks against all next_tracks.
	// Do this processing on batches of blockDim.x size
	for(i=0; i<int(ceilf( ((float) tracks_to_process_size) / blockDim.x)); ++i){
		current_track = blockDim.x * i + threadIdx.x;
		if(current_track < tracks_to_process_size){
			// Store all tracks in sh_tracks
			sh_tracks[threadIdx.x] = tracks[tracks_to_process[current_track]];

			// Calculate chi2
			sh_chi2[threadIdx.x] = trackChi2(sh_tracks[threadIdx.x]);
		}

		__syncthreads();

		// if(sh_tracks[threadIdx.x].hits[0] == 987)
		// 	i = 20;

		// Iterate in all next_tracks
		for(j=0; j<int(ceilf( ((float) tracks_to_process_size) / blockDim.x)); ++j){
			next_track = blockDim.x * j + threadIdx.x;

			if(next_track < tracks_to_process_size){
				// Store all tracks in sh_tracks
				sh_next_tracks[threadIdx.x] = tracks[tracks_to_process[next_track]];

				// Calculate chi2
				sh_next_chi2[threadIdx.x] = trackChi2(sh_tracks[threadIdx.x]);
			}

			__syncthreads();

			// All is loaded, commencing assault!
			for(int k=0; k<BUNCH_POST_TRACKS; ++k){
				next_track = blockDim.x * j + k;

				if(current_track < tracks_to_process_size && next_track < tracks_to_process_size){
					/* Compare all tracks to check uniqueness, based on
					- length
					- chi2

					preferent is a boolean storing this logic. It reads,
				
					TODO: Change preference system by something more civilized
					next_track is preferent if
						it's not current_track,
						its length > current_track . length OR
						(its length == current_track . length AND
						chi2 < current_track . chi2)
					*/
					preferent = current_track!=next_track &&
										(sh_next_tracks[k].hitsNum > sh_tracks[threadIdx.x].hitsNum ||
										(sh_next_tracks[k].hitsNum == sh_tracks[threadIdx.x].hitsNum &&
										sh_next_chi2[k] < sh_chi2[threadIdx.x]));

					// Preference system based solely on chi2
					/*preferent = current_track!=next_track &&
						        sh_next_chi2[k] < sh_chi2[threadIdx.x]; */

					// TODO: Maybe there's a better way...
					if(preferent){
						// Eliminate hits from current_track, based on next_track's
						for(int current_hit=0; current_hit<TRACK_SIZE; ++current_hit){
							for(int next_hit=0; next_hit<TRACK_SIZE; ++next_hit){
								/* apply mask:
								a[i] = 
									(a[i] == b[j]) * -1 +
									(a[i] != b[j]) * a[i]
								*/
								sh_tracks[threadIdx.x].hits[current_hit] =
									(sh_tracks[threadIdx.x].hits[current_hit] == sh_next_tracks[k].hits[next_hit]) * -1 + 
									(sh_tracks[threadIdx.x].hits[current_hit] != sh_next_tracks[k].hits[next_hit]) *
										sh_tracks[threadIdx.x].hits[current_hit];
							}
						}
					}
				}
			}
		}

		if(current_track < tracks_to_process_size){
			// Check how many uniques do we have
			int unique = 0;
			for(int hit=0; hit<TRACK_SIZE; ++hit)
				unique += (sh_tracks[threadIdx.x].hits[hit]!=-1);

			if(((float) unique) / sh_tracks[threadIdx.x].hitsNum > REQUIRED_UNIQUES){
				int current_track_accepted = atomicAdd(&tracks_accepted_size, 1);

				track_indexes[current_track_accepted] = tracks_to_process[current_track];
			}
		}
	}

	__syncthreads();

	if(threadIdx.x==0)
		num_tracks[0] = tracks_accepted_size;
}
