#include "KernelInvoker.cuh"

extern int*   h_no_sensors;
extern int*   h_no_hits;
extern int*   h_sensor_Zs;
extern int*   h_sensor_hitStarts;
extern int*   h_sensor_hitNums;
extern unsigned int* h_hit_IDs;
extern float* h_hit_Xs;
extern float* h_hit_Ys;
extern float* h_hit_Zs;

cudaError_t invokeParallelSearch(
    dim3                         numBlocks,
    dim3                         numThreads,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t>       & solution) {
  // For now, just perform what we did before
  // (backwards compatibility)
  int* h_track_indexes;
  int* num_tracks;
  Track* tracks;

  DEBUG << "Input pointer: " 
    << std::hex << "0x" << (long long int) &(input[0])
    << std::dec << std::endl;

  setHPointersFromInput(const_cast<uint8_t*>(&input[0]), input.size());
  printInfo();

  std::map<int, int> zhit_to_module;
  if (logger::ll.verbosityLevel > 0){
    // map to convert from z of hit to module
    for(int i=0; i<*h_no_sensors; ++i){
      const int z = h_sensor_Zs[i];
      zhit_to_module[z] = i;
    }

    // Some hits z may not correspond to a sensor's,
    // but be close enough
    for(int i=0; i<*h_no_hits; ++i){
      const int z = h_hit_Zs[i];
      if (zhit_to_module.find(z) == zhit_to_module.end()){
        const int sensor = findClosestModule(z, zhit_to_module);
        zhit_to_module[z] = sensor;
      }
    }
  }

  // int* h_prevs, *h_nexts;
  // Histo histo;

  char*  dev_input             = 0;
  int*   dev_num_tracks        = 0;
  int*   dev_track_indexes     = 0;
  Track* dev_tracks            = 0;
  bool*  dev_track_holders     = 0;
  int*   dev_prevs             = 0;
  int*   dev_nexts             = 0;
  int*   dev_tracks_to_process = 0;
  cudaError_t cudaStatus = cudaSuccess;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaCheck( cudaSetDevice(0) );

  // Allocate memory
  // Allocate CPU buffers
  tracks = (Track*) malloc(MAX_TRACKS * sizeof(Track));
  //solution.resize(MAX_TRACKS * sizeof(Track));
  //tracks = (Track*) &(solution[0]);
  num_tracks = (int*) malloc(sizeof(int));

  int* h_prevs = (int*) malloc(h_no_hits[0] * sizeof(int));
  int* h_nexts = (int*) malloc(h_no_hits[0] * sizeof(int));
  bool* h_track_holders = (bool*) malloc(MAX_TRACKS * sizeof(bool));
  h_track_indexes = (int*) malloc(MAX_TRACKS * sizeof(int));

  // Allocate GPU buffers
  cudaCheck(cudaMalloc((void**)&dev_tracks, MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_track_holders, MAX_TRACKS * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_track_indexes, MAX_TRACKS * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_process, MAX_TRACKS * sizeof(int)));

  cudaCheck(cudaMalloc((void**)&dev_prevs, h_no_hits[0] * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_nexts, h_no_hits[0] * sizeof(int)));

  // Copy input file from host memory to GPU buffers
  cudaCheck(cudaMalloc((void**)&dev_input, input.size()));
  cudaCheck(cudaMalloc((void**)&dev_num_tracks, sizeof(int)));

  // memcpys
  cudaCheck(cudaMemcpy(dev_input, &(input[0]), input.size(), cudaMemcpyHostToDevice));

  // Launch a kernel on the GPU with one thread for each element.
  prepareData<<<1, 1>>>(dev_input, dev_prevs, dev_nexts, dev_track_holders);

  // gpuKalman
  DEBUG << "gpuKalman" << std::endl;
  cudaEvent_t start_kalman, start_postprocess, stop;
  float t0, t1, t2;

  cudaEventCreate(&start_kalman);
  cudaEventCreate(&start_postprocess);
  cudaEventCreate(&stop);

  cudaEventRecord(start_kalman, 0 );

  gpuKalman<<<numBlocks, numThreads>>>(dev_tracks, dev_track_holders);

  cudaEventRecord(start_postprocess);


  DEBUG << "postProcess" << std::endl;
  postProcess<<<1, numThreads>>>(dev_tracks, dev_track_holders, dev_track_indexes, dev_num_tracks, dev_tracks_to_process);

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  cudaEventElapsedTime( &t0, start_kalman, start_postprocess );
  cudaEventElapsedTime( &t1, start_postprocess, stop );
  cudaEventElapsedTime( &t2, start_kalman, stop );
  cudaEventDestroy( start_kalman );
  cudaEventDestroy( start_postprocess );
  cudaEventDestroy( stop );

  // Get results
  cudaCheck(cudaMemcpy(h_track_holders, dev_track_holders, MAX_TRACKS * sizeof(bool), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_track_indexes, dev_track_indexes, MAX_TRACKS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(tracks, dev_tracks, MAX_TRACKS * sizeof(Track), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(num_tracks, dev_num_tracks, sizeof(int), cudaMemcpyDeviceToHost));

  // number of tracks after stage#1
  int no_tracks_stage1 = 0;
  for(int i=0; i<h_no_hits[0]; ++i)
    if(h_track_holders[i])
      ++no_tracks_stage1;

  // copy selected track to the solution vector
  if (*num_tracks > 0) {
    solution.resize(*num_tracks * sizeof(Track));
    Track * solutionTracks = (Track*)&solution[0];
    for (size_t i = 0; i != *num_tracks; ++i)
      solutionTracks[i] = tracks[h_track_indexes[i]];
  }

  // print debug info
  if (logger::ll.verbosityLevel > 0){
    for(int i=0; i<num_tracks[0]; ++i){
      printTrack(tracks, h_track_indexes[i], i, zhit_to_module);
    }
  }

  DEBUG << "Processed " << num_tracks[0] << " tracks" << std::endl;

  free(h_prevs);
  free(h_nexts);
  free(h_track_holders);
  free(tracks);
  free(num_tracks);

  return cudaStatus;
}

/**
 * Prints tracks
 * Track #n, length <length>:
 *  <ID> module <module>, x <x>, y <y>, z <z>
 * 
 * @param tracks      
 * @param trackNumber 
 */
void printTrack(Track* tracks, const int trackID,
  const int trackNumber, const std::map<int, int>& zhit_to_module){

  const Track t = tracks[trackID];
  DEBUG << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const int hitNumber = t.hits[i];
    const unsigned int id = h_hit_IDs[hitNumber];
    const float x = h_hit_Xs[hitNumber];
    const float y = h_hit_Ys[hitNumber];
    const float z = h_hit_Zs[hitNumber];
    const int module = zhit_to_module.at((int) z);

    DEBUG << " " << std::setw(8) << id
      << " module " << std::setw(2) << module
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << z << std::endl;
  }

  DEBUG << std::endl;
}

/**
 * The z of the hit may not correspond to any z in the sensors.
 * @param  z              
 * @param  zhit_to_module 
 * @return                sensor number
 */
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module){
  if (zhit_to_module.find(z) != zhit_to_module.end())
    return zhit_to_module.at(z);

  int error = 0;
  while(true){
    error++;
    const int lowerAttempt = z - error;
    const int higherAttempt = z + error;

    if (zhit_to_module.find(lowerAttempt) != zhit_to_module.end()){
      return zhit_to_module.at(lowerAttempt);
    }
    if (zhit_to_module.find(higherAttempt) != zhit_to_module.end()){
      return zhit_to_module.at(higherAttempt);
    }
  }
}

void printOutAllSensorHits(int* prevs, int* nexts){
  DEBUG << "All valid sensor hits: " << std::endl;
  for(int i=0; i<h_no_sensors[0]; ++i){
    for(int j=0; j<h_sensor_hitNums[i]; ++j){
      int hit = h_sensor_hitStarts[i] + j;

      if(nexts[hit] != -1){
        DEBUG << hit << ", " << nexts[hit] << std::endl;
      }
    }
  }
}

void printOutSensorHits(int sensorNumber, int* prevs, int* nexts){
  for(int i=0; i<h_sensor_hitNums[sensorNumber]; ++i){
    int hstart = h_sensor_hitStarts[sensorNumber];

    DEBUG << hstart + i << ": " << prevs[hstart + i] << ", " << nexts[hstart + i] << std::endl;
  }
}

void printInfo() {
  DEBUG << "Read info:" << std::endl
    << " no sensors: " << h_no_sensors[0] << std::endl
    << " no hits: " << h_no_hits[0] << std::endl
    << "First 5 sensors: " << std::endl;

  for (int i=0; i<5; ++i){
    DEBUG << " Zs: " << h_sensor_Zs[i] << std::endl
      << " hitStarts: " << h_sensor_hitStarts[i] << std::endl
      << " hitNums: " << h_sensor_hitNums[i] << std::endl << std::endl;
  }

  DEBUG << "First 5 hits: " << std::endl;

  for (int i=0; i<5; ++i){
    DEBUG << " hit_id: " << h_hit_IDs[i] << std::endl
      << " hit_X: " << h_hit_Xs[i] << std::endl
      << " hit_Y: " << h_hit_Ys[i] << std::endl
      << " hit_Z: " << h_hit_Zs[i] << std::endl << std::endl;
  }
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
