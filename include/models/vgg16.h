#ifndef VGG16_H
#define VGG16_H

#include <opencv2/core.hpp>

#include "module.h"

class VGG16 : public nn::Module
{
public:
  VGG16(size_t input_channels, size_t output_size);

  nn::matrix<float> forward(const cv::Mat& image);
  void backward(const nn::matrix<float>& grad);
};

#endif  // VGG16_H