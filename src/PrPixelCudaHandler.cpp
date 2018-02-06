#include "GpuPixelSearchByTriplet.h"
#include "PrPixelCudaHandler.h"

#include <algorithm>
#include <vector>

using namespace std;

DECLARE_COMPONENT(PrPixelCudaHandler)

void PrPixelCudaHandler::operator() (
    const Batch & batch,
    Alloc         allocResult,
    AllocParam    allocResultParam) {
  // gpuPixelSearchByTriplet handles several events in parallel
  vector<Data> trackCollection;
  gpuPixelSearchByTriplet(batch, trackCollection);

  for (int i = 0, size = trackCollection.size(); i != size; ++i){
    uint8_t * buffer = (uint8_t*)allocResult(i, trackCollection[i].size(), allocResultParam);
    copy(trackCollection[i].begin(), trackCollection[i].end(), buffer);
  }
}
