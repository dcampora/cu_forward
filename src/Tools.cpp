#include "Tools.h"

void preorderByX(std::vector<std::vector<uint8_t>>& input) {
  // Order *all* the input vectors by h_hit_Xs natural order
  // per sensor
  for (int i=0; i<input.size(); ++i) {
    int acc_hitnums = 0;
    auto eventInfo = EventInfo(input[i]);

    for (int j=0; j<eventInfo.numberOfSensors; j++) {
      const int hitnums = eventInfo.sensor_hitNums[j];
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
            swap(a[left], a[right]);
            swap(b[left], b[right]);
            swap(c[left], c[right]);
        }
    }
 
    swap(a[right], a[start]);
    swap(b[right], b[start]);
    swap(c[right], c[start]);
 
    return right;
}

template<typename T>
void swap (T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
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
 * Reads some data from an input file, following the
 * specified format of choice
 *
 * Format expected in file:
 *
 * int funcNameLen
 * char* funcName
 * int dataSize
 * char* data
 */
void readFileIntoVector(const std::string& filename, std::vector<unsigned char>& output){
    // Check if file exists
    if (!fileExists(filename)){
        throw StrException("Error: File " + filename + " does not exist.");
    }

    std::ifstream infile (filename.c_str(), std::ifstream::binary);

    // get size of file
    infile.seekg(0, std::ifstream::end);
    int size = infile.tellg();
    infile.seekg(0);

    // Read format expected:
    //  int funcNameLen
    //  char* funcName
    //  int dataSize
    //  char* data
    int funcNameLen;
    int dataSize;
    std::vector<char> funcName;

    char* pFuncNameLen = (char*) &funcNameLen;
    char* pDataSize = (char*) &dataSize;
    infile.read(pFuncNameLen, sizeof(int));

    funcName.resize(funcNameLen);
    infile.read(&(funcName[0]), funcNameLen);
    infile.read(pDataSize, sizeof(int));

    // read content of infile with a vector
    output.resize(dataSize);
    infile.read ((char*) &(output[0]), dataSize);
    infile.close();
}

std::vector<std::vector<unsigned char>> readFolder (
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
  
  std::vector<std::vector<unsigned char>> input;
  int readFiles = 0;

  for (int i=0; i<fileNumber; ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folderContents[i % folderContents.size()];

    std::vector<unsigned char> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);

    // Check the number of sensors is correct, otherwise ignore it
    auto eventInfo = EventInfo(inputContents);
    if (eventInfo.numberOfSensors == NUMBER_OF_SENSORS) {
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
