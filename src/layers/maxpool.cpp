#include <opencv2/core.hpp>

#include "layers/maxpool.h"

using namespace nn;

Maxpool2D::Maxpool2D(size_t kernel_size, size_t stride, size_t padding)
    : m_params({kernel_size, kernel_size, padding, padding, stride, stride})
{
}

cv::Mat Maxpool2D::forward(const cv::Mat& input)
{
  // TODO: Implement later
}

cv::Mat Maxpool2D::backward(const cv::Mat& grad)
{
  // TODO: Implement later
}