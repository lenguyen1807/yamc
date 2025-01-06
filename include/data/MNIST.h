#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

struct MNISTData
{
  struct image
  {
    // one hot vector for label
    nn::matrix<float> label;
    nn::matrix<float> data;
  };

  explicit MNISTData(const std::string& path);
  std::vector<std::unique_ptr<image>> dataset;
};

#endif  // MNIST_H