#include "../include/Tools.h"

/**
 * @brief Read files into vectors.
 */
void readFileIntoVector(const std::string& filename, std::vector<uint8_t>& output) {
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  // read content of infile with a vector
  output.resize(dataSize);
  infile.read ((char*) &(output[0]), dataSize);
  infile.close();
}

/**
 * @brief Reads a number of events from a folder name.
 */
std::vector<std::vector<uint8_t>> readFolder(
  const std::string& foldername,
  int fileNumber
) {
  std::vector<std::string> folderContents;
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = std::string(ent->d_name);
      if (filename.find(".bin") != std::string::npos) {
        folderContents.push_back(filename);
      }
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      std::cerr << "No binary files found in folder " << foldername << std::endl;
      exit(-1);
    } else {
      std::cout << "Found " << folderContents.size() << " binary files" << std::endl;
    }
  } else {
    std::cerr << "Folder could not be opened" << std::endl;
    exit(-1);
  }
  std::cout << "Requested " << fileNumber << " files" << std::endl;
  std::vector<std::vector<uint8_t>> input;
  int readFiles = 0;
  for (int i=0; i<fileNumber; ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folderContents[i % folderContents.size()];
    std::vector<uint8_t> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);
    // Check the number of modules is correct, otherwise ignore it
    auto eventInfo = EventInfo(inputContents);
    if (eventInfo.numberOfModules == NUMBER_OF_SENSORS) {
      // Make inputContents only the reported size by eventInfo
      inputContents.resize(eventInfo.size);
      input.push_back(inputContents);
    }
    readFiles++;
    if ((readFiles % 100) == 0) {
      std::cout << "." << std::flush;
    }
  }
  std::cout << std::endl << input.size() << " files read" << std::endl << std::endl;
  return input;
}

/**
 * @brief Print statistics from the input files
 */
void statistics(
  const std::vector<std::vector<uint8_t>>& input
) {
  unsigned int max_number_of_hits = 0;
  unsigned int max_number_of_hits_in_module = 0;
  unsigned int average_number_of_hits_in_module = 0;

  for (size_t i=0; i<input.size(); ++i) {
    EventInfo info (input[i]);
    for (size_t j=0; j<info.numberOfModules; ++j) {
      max_number_of_hits_in_module = std::max(max_number_of_hits_in_module, info.module_hitNums[j]);
      average_number_of_hits_in_module += info.module_hitNums[j];
    }
    max_number_of_hits = std::max(max_number_of_hits, info.numberOfHits);
  }
  average_number_of_hits_in_module /= input.size() * 52;

  std::cout << "Statistics on input events:" << std::endl
    << " Max number of hits in event: " << max_number_of_hits << std::endl
    << " Max number of hits in one module: " << max_number_of_hits_in_module << std::endl
    << " Average number of hits in module: " << average_number_of_hits_in_module << std::endl << std::endl;
}

/**
 * @brief Obtains results statistics.
 */
std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = FLT_MAX, max = 0.0f;

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

/**
 * @brief Writes a track in binary format
 * 
 * @details The binary format is per every track:
 *   hitsNum hit0 hit1 hit2 ... (#hitsNum times)
 */
void writeBinaryTrack(
  const unsigned int* hit_IDs,
  const Track& track,
  std::ofstream& outstream
) {
  uint32_t hitsNum = track.hitsNum;
  outstream.write((char*) &hitsNum, sizeof(uint32_t));
  for (int i=0; i<track.hitsNum; ++i) {
    outstream.write((char*) &hit_IDs[track.hits[i]], sizeof(uint32_t));
  }
}

/**
 * Prints tracks
 * Track #n, length <length>:
 *  <ID> module <module>, x <x>, y <y>, z <z>
 * 
 * @param tracks      
 * @param trackNumber 
 */
void printTrack(
  const EventInfo& info,
  Track* tracks,
  const int trackNumber,
  std::ofstream& outstream
) {
  const Track t = tracks[trackNumber];
  outstream << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const int hitNumber = t.hits[i];
    const unsigned int id = info.hit_IDs[hitNumber];
    const float x = info.hit_Xs[hitNumber];
    const float y = info.hit_Ys[hitNumber];

    int module = 0;
    for (int i=0; i<info.numberOfModules; ++i) {
      if (hitNumber >= info.module_hitStarts[i] &&
          hitNumber < info.module_hitStarts[i] + info.module_hitNums[i]) {
        module = i;
      }
    }

    outstream << " " << std::setw(8) << id << " (" << hitNumber << ")"
      << " module " << std::setw(2) << module
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << info.module_Zs[module]
      << std::endl;
  }

  outstream << std::endl;
}

void printOutAllModuleHits(const EventInfo& info, int* prevs, int* nexts) {
  DEBUG << "All valid module hits: " << std::endl;
  for(int i=0; i<info.numberOfModules; ++i){
    for(int j=0; j<info.module_hitNums[i]; ++j){
      int hit = info.module_hitStarts[i] + j;

      if(nexts[hit] != -1){
        DEBUG << hit << ", " << nexts[hit] << std::endl;
      }
    }
  }
}

void printOutModuleHits(const EventInfo& info, int moduleNumber, int* prevs, int* nexts){
  for(int i=0; i<info.module_hitNums[moduleNumber]; ++i){
    int hstart = info.module_hitStarts[moduleNumber];

    DEBUG << hstart + i << ": " << prevs[hstart + i] << ", " << nexts[hstart + i] << std::endl;
  }
}

void printInfo(const EventInfo& info, int numberOfModules, int numberOfHits) {
  numberOfModules = numberOfModules>52 ? 52 : numberOfModules;

  DEBUG << "Read info:" << std::endl
    << " no modules: " << info.numberOfModules << std::endl
    << " no hits: " << info.numberOfHits << std::endl
    << numberOfModules << " modules: " << std::endl;

  for (int i=0; i<numberOfModules; ++i){
    DEBUG << " Zs: " << info.module_Zs[i] << std::endl
      << " hitStarts: " << info.module_hitStarts[i] << std::endl
      << " hitNums: " << info.module_hitNums[i] << std::endl << std::endl;
  }

  DEBUG << numberOfHits << " hits: " << std::endl;

  for (int i=0; i<numberOfHits; ++i){
    DEBUG << " hit_id: " << info.hit_IDs[i] << std::endl
      << " hit_X: " << info.hit_Xs[i] << std::endl
      << " hit_Y: " << info.hit_Ys[i] << std::endl
      // << " hit_Z: " << info.hit_Zs[i] << std::endl
      << std::endl;
  }
}

cudaError_t checkSorting(
  const std::vector<std::vector<uint8_t>>& input,
  unsigned int acc_hits,
  unsigned short* dev_hit_phi,
  const std::vector<unsigned int>& hit_offsets
) {
  // Check sorting
  std::vector<float> hit_phis (acc_hits);
  cudaCheck(cudaMemcpy(hit_phis.data(), &dev_hit_phi[0], acc_hits * sizeof(float), cudaMemcpyDeviceToHost));

  // Check sorting is correct in the resulting phi array
  bool ordered = true;
  for (unsigned int i=0; i<input.size(); ++i) {
    auto info = EventInfo(input[i]);
    // DEBUG << "Event " << i << ":" << std::endl;
    for (unsigned int module=0; module<52; ++module) {
      const unsigned int start_hit = info.module_hitStarts[module];
      const unsigned int num_hits = info.module_hitNums[module];
      float phi = -10.f;
      // DEBUG << "Module " << module << ":";
      for (unsigned int hit_id=start_hit; hit_id<(start_hit + num_hits); ++hit_id) {
        const float hit_phi = hit_phis[hit_offsets[i] + hit_id];
        // DEBUG << " " << hit_phi;
        if (hit_phi < phi) {
          ordered = false;
          // DEBUG << std::endl << hit_phi << " vs " << phi << std::endl;
          break;
        } else {
          phi = hit_phi;
        }
      }
      // DEBUG << std::endl;
      if (!ordered) { break; }
    }
    if (!ordered) { break; }
  }

  DEBUG << (ordered ? "Phi array is properly ordered" : "Phi array is not ordered") << std::endl << std::endl;

  return cudaSuccess;
}
