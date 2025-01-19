#ifndef CIFAR10_H
#define CIFAR10_H

#include <unordered_map>
#include <vector>

#include "matrix.h"
#include "opencv2/core.hpp"

static const std::string CIFAR_DIR = std::string(DATA_SOURCE_DIR) + "cifar-10/";

struct CIFAR10Data
{
  struct image
  {
    cv::Mat data;
    nn::matrix<float> label;  // one hot label
  };

  explicit CIFAR10Data(bool train = true);

  void load_label();
  void load_im(const std::string& path,
               std::vector<std::unique_ptr<image>>& imgs);
  void load_train_im();
  void load_test_im();

  std::unordered_map<std::string, size_t> label_map;
  std::vector<std::unique_ptr<image>> dataset;
};

#endif  // CIFAR10_H