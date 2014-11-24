#include "Tools.cuh"

#include <stdexcept>

// TODO: Remove globals in the short future
int*   h_no_sensors;
int*   h_no_hits;
int*   h_sensor_Zs;
int*   h_sensor_hitStarts;
int*   h_sensor_hitNums;
int*   h_hit_IDs;
float* h_hit_Xs;
float* h_hit_Ys;
int*   h_hit_Zs;

void setHPointersFromInput(uint8_t * input, size_t size){
  uint8_t * end = input + size;

  h_no_sensors       = (int32_t*)input; input += sizeof(int32_t);
  h_no_hits          = (int32_t*)input; input += sizeof(int32_t);
  h_sensor_Zs        = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitStarts = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitNums   = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_hit_IDs          = (int32_t*)input; input += sizeof(int32_t) * *h_no_hits;
  h_hit_Xs           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Ys           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Zs           = (int32_t*)input; input += sizeof(int32_t) * *h_no_hits;

  if (input != end)
    throw std::runtime_error("failed to deserialize event");
}

/**
 * Combine all solutions, in the form:
 *
 * int numberOfEvents
 * per event:
 *     int size
 *
 * per event:
 *     char* output
 *     
 * @param solutions 
 * @param output    
 */
void mergeSolutions(const std::vector<std::vector<char> >& solutions, std::vector<char>& output){
    int numberOfEvents = solutions.size();
    output.resize((numberOfEvents + 1) * sizeof(int));
    char* outputPointer = ((char*) &(output[0]));

    memcpy(outputPointer, &numberOfEvents, sizeof(int));
    outputPointer += sizeof(int);

    int return_size = 0;
    for (int i=0; i<solutions.size(); ++i){
        // Requires lvalue
        int solutions_size = solutions[i].size();
        return_size += solutions_size;
        memcpy(outputPointer, &solutions_size, sizeof(int));
    }
    
    // After resizing, the pointer may change, so recalculate
    output.resize(output.size() + return_size);
    outputPointer = ((char*) &(output[0])) + (numberOfEvents + 1) * sizeof(int);

    for (int i=0; i<solutions.size(); ++i){
        memcpy(outputPointer, &(solutions[i][0]), solutions[i].size());
        outputPointer += solutions[i].size();
    }
}
