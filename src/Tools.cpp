#include "Tools.h"

/**
 * @brief Orders all the input vectors by phi per module
 */
void sort_by_phi(std::vector<std::vector<uint8_t>>& input) {
  for (int i=0; i<input.size(); ++i) {
    auto eventInfo = EventInfo(input[i]);
    // Repurpose hit_Zs and calculate phi instead
    auto phi_container = eventInfo.hit_Zs;
    for (int j=0; j<eventInfo.numberOfModules; ++j) {
      for (int k=0; k<eventInfo.module_hitNums[j]; ++k) {
        const auto index = eventInfo.module_hitStarts[j] + k;
        phi_container[index] = hit_phi(eventInfo.hit_Xs[index], eventInfo.hit_Ys[index], j%2);
      }
    }
    // Calculate the permutation we need
    const auto permutation = sort_permutation(
      phi_container,
      eventInfo.numberOfHits,
      eventInfo.module_hitStarts,
      eventInfo.module_hitNums,
      [] (const float& a, const float& b) { return a<b; }
    );
    // Sort all vectors in place with the calculated permutation vector
    for (int j=0; j<eventInfo.numberOfModules; ++j) {
      const auto start = eventInfo.module_hitStarts[j];
      const auto hitnums = eventInfo.module_hitNums[j];
      apply_permutation_in_place(eventInfo.hit_Xs, permutation.data(), start, hitnums);
      apply_permutation_in_place(eventInfo.hit_Ys, permutation.data(), start, hitnums);
      apply_permutation_in_place(eventInfo.hit_Zs, permutation.data(), start, hitnums);
      apply_permutation_in_place(eventInfo.hit_IDs, permutation.data(), start, hitnums);
    }
  }
}

/**
 * @brief Calculate a single hit phi
 */
float hit_phi(const float x, const float y, const bool odd) {
  const float phi = atan2(y, x);
  const auto greater_than_zero = phi > 0.f;
  return odd*phi +
         !odd*greater_than_zero*phi +
         !odd*!greater_than_zero*(phi + 2*M_PI);
}

bool fileExists(const std::string& name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }   
}

/**
 * @brief Read files into vectors.
 */
void readFileIntoVector(const std::string& filename, std::vector<uint8_t>& output) {
  // Check if file exists
  if (!fileExists(filename)){
    throw StrException("Error: File " + filename + " does not exist.");
  }

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
