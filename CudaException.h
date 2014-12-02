
#ifndef CUDA_EXCEPTION
#define CUDA_EXCEPTION 1

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>

class CudaException : public std::runtime_error {
  public:

    CudaException(cudaError_t err, const char * statement);

    virtual ~CudaException() throw();

  private:

    static std::string makeMessage(cudaError_t err, const char * statement);
};

#endif
