#include "KernelInvoker.cuh"

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
) {
  int eventsToProcess = input.size();

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
  float* dev_best_fits;
  int*   dev_hit_candidates;
  int*   dev_hit_h2_candidates;

  // Choose which GPU to run on, change this on a multi-GPU system.
  const int device_number = 0;

  cudaCheck(cudaSetDevice(device_number));
  cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  cudaDeviceProp* device_properties = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
  cudaGetDeviceProperties(device_properties, 0);

  // Some startup settings
  dim3 numBlocks(eventsToProcess);
  dim3 numThreads(NUMTHREADS_X, 4);

  // Allocate memory
  // Prepare event offset and hit offset
  std::vector<int> event_offsets;
  std::vector<int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  for (int i=0; i<eventsToProcess; ++i) {
    auto info = EventInfo(input[i]);
    const int event_size = input[i].size();
    event_offsets.push_back(acc_size);
    hit_offsets.push_back(acc_hits);
    acc_size += event_size;
    acc_hits += info.numberOfHits;
  }

  // Allocate CPU buffers
  const int atomic_space = NUM_ATOMICS + 1;
  int* atomics = (int*) malloc(eventsToProcess * atomic_space * sizeof(int));  
  int* hit_candidates = (int*) malloc(2 * acc_hits * sizeof(int));

  // Allocate GPU buffers
  cudaCheck(cudaMalloc((void**)&dev_tracks, eventsToProcess * MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_tracklets, acc_hits * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, acc_hits * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, eventsToProcess * TTF_MODULO * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_atomicsStorage, eventsToProcess * atomic_space * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_used, acc_hits * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_input, acc_size));
  cudaCheck(cudaMalloc((void**)&dev_best_fits, eventsToProcess * numThreads.x * MAX_NUMTHREADS_Y * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_hit_candidates, 2 * acc_hits * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_h2_candidates, 2 * acc_hits * sizeof(int)));

  // Copy stuff from host memory to GPU buffers
  cudaCheck(cudaMemcpy(dev_event_offsets, &event_offsets[0], event_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_hit_offsets, &hit_offsets[0], hit_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

  acc_size = 0;
  for (int i=0; i<eventsToProcess; ++i){
    cudaCheck(cudaMemcpy(&dev_input[acc_size], input[i].data(), input[i].size(), cudaMemcpyHostToDevice));
    acc_size += input[i].size();
  }

  // Adding timing
  // Timing calculation
  unsigned int niterations = 3;
  unsigned int nexperiments = 5;

  std::vector<std::vector<float>> time_values {nexperiments};
  std::vector<std::map<std::string, float>> mresults {nexperiments};

  DEBUG << "Now, on your " << device_properties->name << ": searchByTriplet with " << eventsToProcess << " event" << (eventsToProcess>1 ? "s" : "") << std::endl 
	  << " " << nexperiments << " experiments, " << niterations << " iterations" << std::endl;

  if (nexperiments!=1) {
    numThreads.y = 1;
  }

  for (auto i=0; i<nexperiments; ++i) {

    DEBUG << numThreads.x << ", " << numThreads.y << ": " << std::flush;

    for (auto j=0; j<niterations; ++j) {
      // Initialize what we need
      cudaCheck(cudaMemset(dev_hit_used, false, acc_hits * sizeof(bool)));
      cudaCheck(cudaMemset(dev_atomicsStorage, 0, eventsToProcess * atomic_space * sizeof(int)));
      cudaCheck(cudaMemset(dev_hit_candidates, -1, 2 * acc_hits * sizeof(int)));
      cudaCheck(cudaMemset(dev_hit_h2_candidates, -1, 2 * acc_hits * sizeof(int)));

      // Just for debugging purposes
      cudaCheck(cudaMemset(dev_tracks, 0, eventsToProcess * MAX_TRACKS * sizeof(Track)));
      cudaCheck(cudaMemset(dev_tracklets, 0, acc_hits * sizeof(Track)));
      cudaCheck(cudaMemset(dev_tracks_to_follow, 0, eventsToProcess * TTF_MODULO * sizeof(int)));
      cudaCheck(cudaDeviceSynchronize());

      // searchByTriplet
      cudaEvent_t start_searchByTriplet, stop_searchByTriplet;
      float t0;

      cudaEventCreate(&start_searchByTriplet);
      cudaEventCreate(&stop_searchByTriplet);

      cudaEventRecord(start_searchByTriplet, 0 );
      
      searchByTriplet<<<numBlocks, numThreads>>>(dev_tracks, (const char*) dev_input,
        dev_tracks_to_follow, dev_hit_used, dev_atomicsStorage, dev_tracklets,
        dev_weak_tracks, dev_event_offsets, dev_hit_offsets, dev_best_fits,
        dev_hit_candidates, dev_hit_h2_candidates);

      cudaEventRecord( stop_searchByTriplet, 0 );
      cudaEventSynchronize( stop_searchByTriplet );
      cudaEventElapsedTime( &t0, start_searchByTriplet, stop_searchByTriplet );

      cudaEventDestroy( start_searchByTriplet );
      cudaEventDestroy( stop_searchByTriplet );

      cudaCheck( cudaPeekAtLastError() );

      time_values[i].push_back(t0);

      DEBUG << "." << std::flush;
    }
    
    if (nexperiments!=1) {
      numThreads.y *= 2;
    }

    DEBUG << std::endl;
  }

  // Get results
  if (PRINT_SOLUTION) DEBUG << "Number of tracks found per event:" << std::endl << " ";
  cudaCheck(cudaMemcpy(atomics, dev_atomicsStorage, eventsToProcess * atomic_space * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<eventsToProcess; ++i){
    const int numberOfTracks = atomics[i];
    if (PRINT_SOLUTION) DEBUG << numberOfTracks << ", ";

    std::vector<uint8_t> output_track (numberOfTracks * sizeof(Track));
    cudaCheck(cudaMemcpy(output_track.data(), &dev_tracks[i * MAX_TRACKS], numberOfTracks * sizeof(Track), cudaMemcpyDeviceToHost));
    output.push_back(output_track);
  }
  if (PRINT_SOLUTION) DEBUG << std::endl;

  if (PRINT_VERBOSE) {
    // Print solution of all events processed, to results
    for (int i=0; i<eventsToProcess; ++i) {

      // Print to output file with event no.
      const int numberOfTracks = output[i].size() / sizeof(Track);
      Track* tracks_in_solution = (Track*) &(output[i])[0];
      std::ofstream outfile (std::string(RESULTS_FOLDER) + std::string("/") + std::to_string(i) + std::string(".out"));
      for(int j=0; j<numberOfTracks; ++j){
        printTrack(EventInfo(input[i]), tracks_in_solution, j, outfile);
      }
      outfile.close();
    }
  }

  if (PRINT_BINARY) {
    std::cout << "Printing binary solution" << std::endl;
    for (int i=0; i<eventsToProcess; ++i) {
      const int numberOfTracks = output[i].size() / sizeof(Track);
      Track* tracks_in_solution = (Track*) &(output[i])[0];

      std::ofstream outfile (std::string(RESULTS_FOLDER) + std::string("/tracks_") + std::to_string(i) + std::string(".bin"), std::ios::binary);
      outfile.write((char*) &numberOfTracks, sizeof(int32_t));
      for(int j=0; j<numberOfTracks; ++j){
        writeBinaryTrack(EventInfo(input[i]), tracks_in_solution[j], outfile);
      }
      outfile.close();

      if ((i%100) == 0) {
        std::cout << "." << std::flush;
      }
    }
    std::cout << std::endl;
  }

  DEBUG << std::endl << "Time averages:" << std::endl;
  int exp = 1;
  for (auto i=0; i<nexperiments; ++i){
    mresults[i] = calcResults(time_values[i]);
    DEBUG << " nthreads (" << NUMTHREADS_X << ", " << (nexperiments==1 ? numThreads.y : exp) <<  "): "
      << eventsToProcess / (mresults[i]["mean"] * 0.001) << " events/s, "
      << mresults[i]["mean"] << " ms (std dev " << mresults[i]["deviation"] << ")" << std::endl;

    exp *= 2;
  }

  free(atomics);

  return cudaSuccess;
}
