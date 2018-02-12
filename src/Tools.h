#pragma once

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Common.h"
#include <algorithm>

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