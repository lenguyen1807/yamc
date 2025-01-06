#ifndef ALEXNET_H
#define ALEXNET_H

#include <opencv2/core.hpp>

#include "matrix.h"
#include "module.h"

class AlexNet : public nn::Module
{
public:
  /*
  - We will use base implementation of AlexNet with 10 class output
  - https://en.wikipedia.org/wiki/AlexNet
  */
  AlexNet();

  // Now declare forward and backward function
  nn::matrix<float> forward(const cv::Mat& input);
  void backward(const nn::matrix<float>& grad);
};

#endif  // ALEXNET_H