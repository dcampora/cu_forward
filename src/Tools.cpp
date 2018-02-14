#include "Tools.h"

void preorderByX(std::vector<std::vector<uint8_t>>& input) {
  // Order *all* the input vectors by h_hit_Xs natural order
  // per module
  for (int i=0; i<input.size(); ++i) {
    int acc_hitnums = 0;
    auto eventInfo = EventInfo(input[i]);

    for (int j=0; j<eventInfo.numberOfModules; j++) {
      const int hitnums = eventInfo.module_hitNums[j];
      quicksort(eventInfo.hit_Xs, eventInfo.hit_Ys, eventInfo.hit_IDs, acc_hitnums, acc_hitnums + hitnums - 1);
      acc_hitnums += hitnums;
    }
  }
}

void quicksort (float* a, float* b, unsigned int* c, int start, int end) {
    if (start < end) {
        const int pivot = divide(a, b, c, start, end);
        quicksort(a, b, c, start, pivot - 1);
        quicksort(a, b, c, pivot + 1, end);
    }
}

int divide (float* a, float* b, unsigned int* c, int start, int end) {
    int left;
    int right;
    float pivot;
 
    pivot = a[start];
    left = start;
    right = end;
 
    while (left < right) {
        while (a[right] > pivot) {
            right--;
        }
 
        while ((left < right) && (a[left] <= pivot)) {
            left++;
        }
 
        if (left < right) {
            std::swap(a[left], a[right]);
            std::swap(b[left], b[right]);
            std::swap(c[left], c[right]);
        }
    }
 
    std::swap(a[right], a[start]);
    std::swap(b[right], b[start]);
    std::swap(c[right], c[start]);
 
    return right;
}

bool fileExists (const std::string& name) {
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
