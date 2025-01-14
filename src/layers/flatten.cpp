#include "layers/conv.h"
#include "layers/flatten.h"

using namespace nn;

matrix<float> Flatten::forward_im(const cv::Mat& im)
{
  // store information for backward pass
  m_channels = im.channels();
  m_height = im.rows;
  m_width = im.cols;

  // flatten to 1 column vector
  auto result = Conv2D::reshape_im2mat(im);
  // Change this to make this become 1D vector
  result.rows = m_channels * m_height * m_width;
  result.cols = 1;

  return result;
}

// Just a copy from reshape_mat2im function
cv::Mat Flatten::backward_im(const matrix<float>& mat)
{
  cv::Mat output(m_height, m_width, CV_32FC(m_channels));
#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < m_channels; c++) {
    for (size_t h = 0; h < m_height; h++) {
      for (size_t w = 0; w < m_width; w++) {
        size_t idx = (c * m_height + h) * m_width + w;
        if (m_channels == 1) {
          output.at<float>(h, w) = mat.data[idx];
        } else {
          output.ptr<float>(h, w)[c] = mat.data[idx];
        }
      }
    }
  }

  return output.clone();
}