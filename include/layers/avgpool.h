#ifndef AVGPOOL_H
#define AVGPOOL_H

#include <opencv2/core/mat.hpp>

#include "helper.h"
#include "layer.h"

namespace nn
{
class AvgPool2D : public Layer<float>
{
public:
  AvgPool2D(size_t kernel_size, size_t stride, size_t padding = 0);

  cv::Mat forward(const cv::Mat& input);
  cv::Mat backward(const cv::Mat& grad);

private:
  ConvParams m_params;
  cv::Mat m_input;
};
};  // namespace nn

#endif  // AVGPOOL_H