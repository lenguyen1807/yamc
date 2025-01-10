#ifndef LENET5_H
#define LENET5_H

#include <opencv2/core.hpp>

#include "module.h"

class LeNet5 : public nn::Module
{
public:
  LeNet5(size_t input_channels, size_t output_size);

  nn::matrix<float> forward(const cv::Mat& image);
  void backward(const nn::matrix<float>& grad);

private:
};

#endif  // LENET5_H