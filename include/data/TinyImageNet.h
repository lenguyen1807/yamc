#ifndef TINYIMAGENET_H
#define TINYIMAGENET_H

#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>

#include "matrix.h"

static const std::string TINY_IMAGENET_DIR =
    std::string(DATA_SOURCE_DIR) + "tiny-imagenet-200/";

struct TinyImageNetData
{
  struct image
  {
    cv::Mat data;
    nn::matrix<float> label;  // one hot label
  };

  explicit TinyImageNetData(bool train = true);

  void load_label();
  void load_test_label();
  void load_train_im();
  void load_test_im();

  std::unordered_map<std::string, size_t> test_labels;
  std::unordered_map<std::string, size_t> label_map;
  std::vector<std::unique_ptr<image>> dataset;
};

#endif  // TINYIMAGENET_H