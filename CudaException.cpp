#include "CudaException.h"

#include <sstream>

using namespace std;

CudaException::CudaException(cudaError_t err, const char * statement) :
    runtime_error(makeMessage(err, statement)) {
}

CudaException::~CudaException() throw() {
}

string CudaException::makeMessage(cudaError_t err, const char * statement) {
  ostringstream str;
  str << "Failed to run: '" << statement << "'. " << cudaGetErrorString(err);
  return str.str();
}
