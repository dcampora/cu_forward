
#ifndef GPUPIXELSEARCHBYTRIPLET
#define GPUPIXELSEARCHBYTRIPLET 1

#include "FileStdLogger.h"
#include "Tools.cuh"
#include "KernelInvoker.cuh"

#include <stdint.h>

int independent_execute(
    const std::vector<std::vector<uint8_t> > & input,
    std::vector<std::vector<uint8_t> > & output);

void independent_post_execute(const std::vector<std::vector<uint8_t> > & output);

int gpuPixelSearchByTriplet(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output);

/**
 * Common entrypoint for Gaudi and non-Gaudi
 * @param input  
 * @param output 
 * @param logger 
 */
int gpuPixelSearchByTripletInvocation(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output,
    std::ostream      & logger);

#endif
