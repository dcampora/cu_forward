
#include "GpuPixelSearchByTriplet.h"

int independent_execute(
    const std::vector<std::vector<uint8_t> > & input,
    std::vector<std::vector<uint8_t> > & output) {

  std::vector<const std::vector<uint8_t>* > converted_input;
  converted_input.resize(input.size());

  for (int i=0; i<input.size(); ++i) {
    converted_input[i] = &(input[i]);
  }

  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = 3;

  return gpuPixelSearchByTripletInvocation(converted_input, output);
}

void independent_post_execute(const std::vector<std::vector<uint8_t> > & output) {
    DEBUG << "post_execute invoked" << std::endl;
    DEBUG << "Size of output: " << output.size() << " B" << std::endl;
}

int gpuPixelSearchByTriplet(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {

  // Silent execution
  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = 0;
  return gpuPixelSearchByTripletInvocation(input, output);
}

/**
 * Common entrypoint for Gaudi and non-Gaudi
 * @param input  
 * @param output 
 */
int gpuPixelSearchByTripletInvocation(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {
  DEBUG << "Invoking gpuPixelSearchByTriplet with " << input.size() << " events" << std::endl;

  // Define how many blocks / threads we need to deal with numberOfEvents

  // For each event, we will execute 52 blocks and 32 threads.
  // Call a kernel for each event, let CUDA engine decide when to issue the kernels.
  dim3 numBlocks(1), numThreads(32);

  // In principle, each execution will return a different output
  output.resize(input.size());

  // This should be done in streams (non-blocking)
  for (int i=0; i<input.size(); ++i)
    cudaCheck(invokeParallelSearch(numBlocks, numThreads, *(input[i]), output[i]));

  cudaCheck(cudaDeviceReset());

  // Deprecated:
  // Merge all solutions!
  // logger << "Merging solutions..." << endl;
  // mergeSolutions(solutions, output);

  return 0;
}
