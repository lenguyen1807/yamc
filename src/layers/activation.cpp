#include <stdexcept>

#include "layers/activation.h"

namespace nn
{
/* ReLU */
matrix<float> ReLU::forward(const matrix<float>& input)
{
  m_input = input;
  return matrix<float>::where(
      m_input > 0.0f, m_input, matrix<float>::values_like(0.0f, m_input));
}

matrix<float> ReLU::backward(const matrix<float>& grad)
{
  return matrix<float>::where(
      m_input > 0.0f, grad, matrix<float>::values_like(0.0f, m_input));
}

cv::Mat ReLU::forward(const cv::Mat& input)
{
  m_im = input.clone();

  // We have to traverse image and apply ReLU for each pixel
  // I know this implementation is suck but we have no choice :<

  cv::Mat result =
      cv::Mat::zeros(input.rows, input.cols, CV_32FC(input.channels()));

#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < input.channels(); c++) {
    for (size_t h = 0; h < input.rows; h++) {
      for (size_t w = 0; w < input.cols; w++) {
        if (input.channels() == 1) {
          result.at<float>(h, w) = std::max(input.at<float>(h, w), 0.f);
        } else {
          result.ptr<float>(h, w)[c] = std::max(input.ptr<float>(h, w)[c], 0.f);
        }
      }
    }
  }

  return result;
}

cv::Mat ReLU::backward(const cv::Mat& grad)
{
  if (grad.rows != m_im.rows || grad.cols != m_im.cols
      || grad.channels() != m_im.channels())
  {
    throw std::invalid_argument("gradient should be equal to input image");
  }

  cv::Mat result =
      cv::Mat::zeros(grad.rows, grad.cols, CV_32FC(grad.channels()));

#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < grad.channels(); c++) {
    for (size_t h = 0; h < grad.rows; h++) {
      for (size_t w = 0; w < grad.cols; w++) {
        if (grad.channels() == 1) {
          result.at<float>(h, w) =
              grad.at<float>(h, w) * std::max(m_im.at<float>(h, w), 0.f);
        } else {
          result.ptr<float>(h, w)[c] = grad.ptr<float>(h, w)[c]
              * std::max(m_im.ptr<float>(h, w)[c], 0.f);
        }
      }
    }
  }

  return result;
}

/* Softmax */
matrix<float> Softmax::forward(const matrix<float>& input)
{
  float max_input = matrix<float>::max(input);
  matrix<float> e_x =
      (input - max_input).apply([](float x) { return std::expf(x); });

  // This is a trick because I'm too lazy to implement / a scalar
  return e_x % (1.0f / (e_x.reduce_sum()));
}

matrix<float> Softmax::backward(const matrix<float>& grad)
{
  // You should left softmax derivative for loss function
  return grad;
}

matrix<float> LeakyReLU::forward(const matrix<float>& input)
{
  m_input = input;
  return matrix<float>::where(m_input > 0.0f, m_input, m_input % m_slope);
}

matrix<float> LeakyReLU::backward(const matrix<float>& grad)
{
  return matrix<float>::where(
      m_input > 0.0f, grad, matrix<float>::values_like(m_slope, m_input));
}
};  // namespace nn