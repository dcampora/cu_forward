#pragma once

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

bool fileExists (const std::string& name);
void readFileIntoVector(const std::string& foldername, std::vector<unsigned char> & output);

std::vector<std::vector<unsigned char>> readFolder (
  const std::string& foldername,
  int fileNumber
);

// A non-efficient implementation that does what I need
void preorderByX(std::vector<std::vector<uint8_t>>& input);
void quicksort(float* a, float* b, unsigned int* c, int start, int end);
int divide(float* a, float* b, unsigned int* c, int first, int last);
template<typename T> void swap(T& a, T& b);
