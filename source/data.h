#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>

#include "matrix.h"

struct image
{
  // one hot vector for label
  nn::matrix<double> label;
  nn::matrix<double> data;
};

struct MNISTData
{
  explicit MNISTData(const std::string& path);
  std::vector<std::unique_ptr<image>> dataset;
};

#endif  // DATA_H