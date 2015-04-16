#include "Tools.cuh"

#include <stdexcept>

// TODO: Remove globals in the short future
int*   h_no_sensors;
int*   h_no_hits;
int*   h_sensor_Zs;
int*   h_sensor_hitStarts;
int*   h_sensor_hitNums;
unsigned int* h_hit_IDs;
float* h_hit_Xs;
float* h_hit_Ys;
float* h_hit_Zs;

void setHPointersFromInput(uint8_t * input, size_t size){
  uint8_t * end = input + size;

  h_no_sensors       = (int32_t*)input; input += sizeof(int32_t);
  h_no_hits          = (int32_t*)input; input += sizeof(int32_t);
  h_sensor_Zs        = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitStarts = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitNums   = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * *h_no_hits;
  h_hit_Xs           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Ys           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Zs           = (float*)  input; input += sizeof(float)   * *h_no_hits;

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

std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = float_max(), max = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min) min = seconds;
        if (seconds > max) max = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min;
    results["max"] = max;

    return results;
}

float float_max() {
    const int value = 0x7f800000;
    const float* const fvalue = (const float*) &value;
    return *(float*)& fvalue[0];
}

/**
 * Quicksort.
 * @param a - The array to be sorted.
 * @param first - The start of the sequence to be sorted.
 * @param last - The end of the sequence to be sorted.
*/
void quickSort(float* a, float* b, float* c, unsigned int* d, int first, int last) 
{
    int pivotElement;
 
    if(first < last)
    {
        pivotElement = pivot(a, b, c, d, first, last);
        quickSort(a, b, c, d, first, pivotElement-1);
        quickSort(a, b, c, d, pivotElement+1, last);
    }
}
 
/**
 * Find and return the index of pivot element.
 * @param a - The array.
 * @param first - The start of the sequence.
 * @param last - The end of the sequence.
 * @return - the pivot element
*/
int pivot(float* a, float* b, float* c, unsigned int* d, int first, int last)
{
    int p = first;
    float pivotElement = a[first];
 
    for(int i = first+1; i <= last; i++)
    {
        if(a[i] < pivotElement)
        {
            p++;
            swap(a[i], a[p]);
            swap(b[i], b[p]);
            swap(c[i], c[p]);
            swap(d[i], d[p]);
        }
    }
 
    swap(a[p], a[first]);
    swap(b[p], b[first]);
    swap(c[p], c[first]);
    swap(d[p], d[first]);
 
    return p;
}


/**
 * Swap the parameters.
 * @param a - The first parameter.
 * @param b - The second parameter.
*/
template<typename T>
void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}
