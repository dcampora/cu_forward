/**
 *      CUDA Forward project
 *      
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
 *
 *      June, 2014 - February, 2018
 *      CERN
 */

#include <iostream>
#include <string>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "Common.h"
#include "Logger.h"
#include "Tools.h"
#include "cuda_runtime.h"

extern cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
);

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0] << " <folder containing .bin files> <number of files to process>"
    << std::endl;
}

int main(int argc, char *argv[])
{
    std::string foldername;
    std::vector<std::string> folderContents;
    std::vector<std::vector<unsigned char>> input;
    int fileNumber;

    // Get params (getopt independent - Compatible with Windows)
    if (argc < 3){
        printUsage(argv);
        return 0;
    }
    foldername = std::string(argv[1]);
    fileNumber = atoi(argv[2]);

    // Check how many files were specified and
    // call the entrypoint with the suggested format
    if(foldername.empty()){
        std::cerr << "No folder specified" << std::endl;
        printUsage(argv);
        return -1;
    }

    // Read folder contents
    input = readFolder(foldername, fileNumber);

    // Call offloaded algo
    std::vector<std::vector<unsigned char>> output;

    // Set verbosity to max
    std::cout << std::fixed << std::setprecision(2);
    logger::ll.verbosityLevel = 3;

    // Show some statistics
    statistics(input);

    // Preorder events by X
    preorderByX(input);

    // Attempt to execute all in one go
    cudaCheck(invokeParallelSearch(input, output));
    cudaCheck(cudaDeviceReset());

    return 0;
}
