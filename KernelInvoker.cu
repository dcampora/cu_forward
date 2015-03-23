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
    const int startingEvent,
    const int eventsToProcess,
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {

  // DEBUG << "Input pointer: " 
  //   << std::hex << "0x" << (long long int) &(input[0])
  //   << std::dec << std::endl;
  
  const std::vector<uint8_t>* startingEvent_input = input[startingEvent];
  setHPointersFromInput((uint8_t*) &(*startingEvent_input)[0], startingEvent_input->size());
  // printInfo();

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

  // int j = 0;
  // for (int z=0; z<input.size(); ++z){
  //   setHPointersFromInput((uint8_t*) &(*input[z])[0], input[z]->size());
  //   for (int i=0; i<*h_no_sensors; ++i) {
  //     if (h_sensor_hitNums[i] > j){
  //       j = h_sensor_hitNums[i];
  //     }
  //   }
  // }
  // DEBUG << "Max hitnums in event: " << j << std::endl;

  // int* h_prevs, *h_nexts;
  // Histo histo;
  Track* dev_tracks;
  char*  dev_input;
  int*   dev_tracks_to_follow;
  bool*  dev_hit_used;
  int*   dev_atomicsStorage;
  Track* dev_tracklets;
  int*   dev_weak_tracks;
  int*   dev_event_offsets;
  int*   dev_hit_offsets;

  // Choose which GPU to run on, change this on a multi-GPU system.
  const int device_number = 0;
  cudaCheck(cudaSetDevice(device_number));
  cudaDeviceProp* device_properties = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
  cudaGetDeviceProperties(device_properties, 0);

  // Some startup settings
  dim3 numBlocks(eventsToProcess);
  dim3 numThreads(64);
  cudaFuncSetCacheConfig(searchByTriplet, cudaFuncCachePreferShared);

  // Allocate memory
  // Allocate CPU buffers
  const int num_atomics = 5;
  int* atomics = (int*) malloc(eventsToProcess * num_atomics * sizeof(int));

  // Prepare event offset and hit offset
  std::vector<int> event_offsets;
  std::vector<int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  for (int i=0; i<eventsToProcess; ++i){
    EventBeginning* event = (EventBeginning*) &(*(input[startingEvent + i]))[0];
    const int event_size = input[startingEvent + i]->size();

    event_offsets.push_back(acc_size);
    hit_offsets.push_back(acc_hits);

    acc_size += event_size;
    acc_hits += event->numberOfHits;
  }

  // Allocate GPU buffers
  cudaCheck(cudaMalloc((void**)&dev_tracks, eventsToProcess * MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_tracklets, eventsToProcess * MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, eventsToProcess * MAX_TRACKS * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, eventsToProcess * MAX_TRACKS * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_atomicsStorage, eventsToProcess * num_atomics * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_used, acc_hits * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_input, acc_size));

  // Copy stuff from host memory to GPU buffers
  cudaCheck(cudaMemcpy(dev_event_offsets, &event_offsets[0], event_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_hit_offsets, &hit_offsets[0], hit_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

  acc_size = 0;
  for (int i=0; i<eventsToProcess; ++i){
    cudaCheck(cudaMemcpy(&dev_input[acc_size], &(*(input[startingEvent + i]))[0], input[startingEvent + i]->size(), cudaMemcpyHostToDevice));
    acc_size += input[startingEvent + i]->size();
  }

  // Initialize what we need
  cudaCheck(cudaMemset(dev_hit_used, false, acc_hits * sizeof(bool)));
  cudaCheck(cudaMemset(dev_atomicsStorage, 0, eventsToProcess * num_atomics * sizeof(int)));

  // searchByTriplet
  DEBUG << "Now, on your " << device_properties->name << ": searchByTriplet with " << eventsToProcess << " event"
    << (eventsToProcess>1 ? "s" : "") << "..." << std::endl;
  cudaEvent_t start_searchByTriplet, stop_searchByTriplet;
  float t0;

  cudaEventCreate(&start_searchByTriplet);
  cudaEventCreate(&stop_searchByTriplet);

  cudaEventRecord(start_searchByTriplet, 0 );
  
  // Dynamic allocation - , 3 * numThreads.x * sizeof(float)
  searchByTriplet<<<numBlocks, numThreads>>>(dev_tracks, (const char*) dev_input, dev_tracks_to_follow,
    dev_hit_used, dev_atomicsStorage, dev_tracklets, dev_weak_tracks, dev_event_offsets, dev_hit_offsets);

  cudaEventRecord( stop_searchByTriplet, 0 );
  cudaEventSynchronize( stop_searchByTriplet );
  cudaEventElapsedTime( &t0, start_searchByTriplet, stop_searchByTriplet );

  cudaEventDestroy( start_searchByTriplet );
  cudaEventDestroy( stop_searchByTriplet );

  cudaCheck( cudaPeekAtLastError() );

  DEBUG << "Done!" << std::endl;

  // Get results
  DEBUG << "Number of tracks found per event:" << std::endl << " ";
  cudaCheck(cudaMemcpy(atomics, dev_atomicsStorage, eventsToProcess * num_atomics * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<eventsToProcess; ++i){
    const int numberOfTracks = atomics[i];
    DEBUG << numberOfTracks << ", ";
    
    output[startingEvent + i].resize(numberOfTracks * sizeof(Track));
    cudaCheck(cudaMemcpy(&(output[startingEvent + i])[0], &dev_tracks[i * MAX_TRACKS], numberOfTracks * sizeof(Track), cudaMemcpyDeviceToHost));
  }
  DEBUG << std::endl;

  // Print info about the solution
  // const int numberOfTracks = output[0].size() / sizeof(Track);
  // Track* tracks_in_solution = (Track*) &(output[0])[0];
  // print debug info
  // if (logger::ll.verbosityLevel > 0){
  //   for(int i=0; i<numberOfTracks; ++i){
  //     printTrack(tracks_in_solution, i, zhit_to_module);
  //   }
  // }
  // DEBUG << "Got " << numberOfTracks << " tracks" << std::endl;
  
  DEBUG << "It took " << t0 << " milliseconds." << std::endl;

  free(atomics);

  return cudaSuccess;
}

/**
 * Prints tracks
 * Track #n, length <length>:
 *  <ID> module <module>, x <x>, y <y>, z <z>
 * 
 * @param tracks      
 * @param trackNumber 
 */
void printTrack(Track* tracks, const int trackNumber, const std::map<int, int>& zhit_to_module){
  const Track t = tracks[trackNumber];
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
