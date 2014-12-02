
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

  return gpuPixelSearchByTripletInvocation(converted_input, output, std::cout);
}

void independent_post_execute(const std::vector<std::vector<uint8_t> > & output) {
    std::cout << "post_execute invoked" << std::endl;
    std::cout << "Size of output: " << output.size() << " B" << std::endl;
}

int gpuPixelSearchByTriplet(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {

  FileStdLogger discardStream;
  VoidLogger logger(&discardStream);

  // std::vector<std::vector<uint8_t> > converted_input;
  // convert_input_swap(converted_input, input);

  // Silent execution
  return gpuPixelSearchByTripletInvocation(input, output, discardStream);

  // convert_input_swap_reverse(converted_input, input);
}

/**
 * Common entrypoint for Gaudi and non-Gaudi
 * @param input  
 * @param output 
 * @param logger
 */
int gpuPixelSearchByTripletInvocation(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output,
    std::ostream      & logger) {
  logger << "Invoking gpuPixelSearchByTriplet with " << input.size() << " events" << std::endl;

  // Define how many blocks / threads we need to deal with numberOfEvents

  // For each event, we will execute 48 blocks and 32 threads.
  // Call a kernel for each event, let CUDA engine decide when to issue the kernels.
  dim3 numBlocks(46), numThreads(32);

  // In principle, each execution will return a different output
  output.resize(input.size());

  // This should be done in streams (non-blocking)
  for (int i=0; i<input.size(); ++i)
    cudaCheck(invokeParallelSearch(numBlocks, numThreads, *(input[i]), output[i], logger));

  cudaCheck(cudaDeviceReset());

  // Deprecated:
  // Merge all solutions!
  // logger << "Merging solutions..." << endl;
  // mergeSolutions(solutions, output);

  return 0;
}
