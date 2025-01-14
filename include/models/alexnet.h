#ifndef ALEXNET_H
#define ALEXNET_H

#include <opencv2/core.hpp>

#include "matrix.h"
#include "module.h"

class AlexNet : public nn::Module
{
public:
  /*
  - We will use small implementation of AlexNet with 10 class output and 32x32
  image
  */
  AlexNet(size_t input_channels, size_t output_size);

  // Now declare forward and backward function
  nn::matrix<float> forward(const cv::Mat& input);
  void backward(const nn::matrix<float>& grad);
};

#endif  // ALEXNET_H