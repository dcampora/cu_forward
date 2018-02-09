#include "Tools.cuh"

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
  const EventInfo& info,
  const Track& track,
  std::ofstream& outstream
) {
  outstream.write((char*) &track.hitsNum, sizeof(uint32_t));
  for (int i=0; i<track.hitsNum; ++i) {
    outstream.write((char*) &info.hit_IDs[track.hits[i]], sizeof(uint32_t));
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
    for (int i=0; i<info.numberOfSensors; ++i) {
      if (hitNumber >= info.sensor_hitStarts[i] &&
          hitNumber < info.sensor_hitStarts[i] + info.sensor_hitNums[i]) {
        module = i;
      }
    }

    outstream << " " << std::setw(8) << id << " (" << hitNumber << ")"
      << " module " << std::setw(2) << module
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << info.sensor_Zs[module]
      << std::endl;
  }

  outstream << std::endl;
}

void printOutAllSensorHits(const EventInfo& info, int* prevs, int* nexts) {
  DEBUG << "All valid sensor hits: " << std::endl;
  for(int i=0; i<info.numberOfSensors; ++i){
    for(int j=0; j<info.sensor_hitNums[i]; ++j){
      int hit = info.sensor_hitStarts[i] + j;

      if(nexts[hit] != -1){
        DEBUG << hit << ", " << nexts[hit] << std::endl;
      }
    }
  }
}

void printOutSensorHits(const EventInfo& info, int sensorNumber, int* prevs, int* nexts){
  for(int i=0; i<info.sensor_hitNums[sensorNumber]; ++i){
    int hstart = info.sensor_hitStarts[sensorNumber];

    DEBUG << hstart + i << ": " << prevs[hstart + i] << ", " << nexts[hstart + i] << std::endl;
  }
}

void printInfo(const EventInfo& info, int numberOfSensors, int numberOfHits) {
  numberOfSensors = numberOfSensors>52 ? 52 : numberOfSensors;

  DEBUG << "Read info:" << std::endl
    << " no sensors: " << info.numberOfSensors << std::endl
    << " no hits: " << info.numberOfHits << std::endl
    << numberOfSensors << " sensors: " << std::endl;

  for (int i=0; i<numberOfSensors; ++i){
    DEBUG << " Zs: " << info.sensor_Zs[i] << std::endl
      << " hitStarts: " << info.sensor_hitStarts[i] << std::endl
      << " hitNums: " << info.sensor_hitNums[i] << std::endl << std::endl;
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
