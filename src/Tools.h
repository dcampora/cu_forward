#pragma once

#include <dirent.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include "Common.h"

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};

bool fileExists(const std::string& name);

void readFileIntoVector(
  const std::string& foldername,
  std::vector<uint8_t> & output
);

std::vector<std::vector<uint8_t>> readFolder(
  const std::string& foldername,
  int fileNumber
);

void statistics(
  const std::vector<std::vector<uint8_t>>& input
);

void sort_by_phi(
  std::vector<std::vector<uint8_t>>& input
);

float hit_phi(
  const float x,
  const float y,
  const bool odd
);

template <class T, class Compare>
std::vector<size_t> sort_permutation(
  T* vec,
  size_t size,
  uint32_t* module_hitStarts,
  uint32_t* module_hitNums,
  Compare compare
) {
  std::vector<size_t> p (size);
  std::iota(p.begin(), p.end(), 0);
  for (int i=0; i<52; ++i) {
    std::stable_sort(p.begin() + module_hitStarts[i], p.begin() + module_hitStarts[i] + module_hitNums[i],
      [&] (size_t i, size_t j) { return compare(vec[i], vec[j]); });
  }
  return p;
}

template <typename T>
void apply_permutation_in_place(
  T* v,
  const size_t* p,
  size_t start,
  size_t numhits
) {
  std::vector<bool> done(numhits);
  for (std::size_t i=0; i<numhits; ++i) {
    if (done[i]) {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = start + i;
    std::size_t j = p[start + i];
    while ((start + i) != j) {
      std::swap(v[prev_j], v[j]);
      done[j - start] = true;
      prev_j = j;
      j = p[j];
    }
  }
}
