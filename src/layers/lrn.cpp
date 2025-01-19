#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "layers/lrn.h"

using namespace nn;

LocalResponseNorm::LocalResponseNorm(float alpha,
                                     float beta,
                                     float k,
                                     size_t local_size)
    : m_alpha(alpha)
    , m_beta(beta)
    , m_k(k)
    , m_size(local_size)
{
}

cv::Mat LocalResponseNorm::forward(const cv::Mat& input)
{
  m_im_channels = input.channels();
  m_im_height = input.rows;
  m_im_width = input.cols;

  if (!m_im.empty()) {
    m_im.release();
  }
  m_im = input.clone();

  m_square_sum.clear();
  m_norm_factor.clear();

  std::vector<cv::Mat> input_im_channels;
  cv::split(input, input_im_channels);

  for (int i = 0; i < input_im_channels.size(); ++i) {
    int start_idx = std::max(0, i - (int)m_size / 2);
    int end_idx = std::min((int)m_im_channels - 1, i + (int)m_size / 2);

    cv::Mat result(input_im_channels[i].size(), CV_32F);

    // Compute square sum
    for (size_t j = start_idx; j < end_idx; ++j) {
      cv::Mat squared;
      cv::multiply(input_im_channels[j], input_im_channels[j], squared);
      cv::accumulate(result, squared);
    }

    // Store square sum for backward
    m_square_sum.push_back(result.clone());

    // Compute normalization matrix
    // alpha * square_sum
    cv::multiply(result, m_alpha, result);
    // k + (alpha * square_sum)
    result += m_k;
    // (k + (alpha * square_sum))^beta
    cv::pow(result, m_beta, result);

    // Store normalization for backward
    m_norm_factor.push_back(result.clone());

    // Divide normalization (input / normalize)
    cv::divide(input_im_channels[i], result, input_im_channels[i]);
  }

  cv::Mat result;
  cv::merge(input_im_channels, result);

  return result.clone();
}

cv::Mat LocalResponseNorm::backward(const cv::Mat& grad)
{
  // TODO:
}