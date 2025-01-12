#include <opencv2/core.hpp>

#include "layers/avgpool.h"
#include "layers/conv.h"

using namespace nn;

AvgPool2D::AvgPool2D(size_t kernel_size, size_t stride, size_t padding)
    : m_params({kernel_size, kernel_size, padding, padding, stride, stride})
{
}

cv::Mat AvgPool2D::forward(const cv::Mat& input)
{
  m_im = input.clone();

  // Calculate output size
  size_t output_h, output_w;
  std::tie(output_h, output_w) =
      Conv2D::calculate_output_size(input.rows, input.cols, m_params);

  /* Then we find average value for each columns in each channels for image
   * (through im2col) (seperate by rows). The easiest implementation is
   * splitting image to each channel, then process each channel seperately so I
   * won't care about the index so much */
  std::vector<cv::Mat> input_splits;
  cv::split(m_im, input_splits);
  matrix<float> result;
  m_input_cols.clear();

  for (const auto& img : input_splits) {
    auto im_col = Conv2D::im2col(img,
                                 m_params.ker_h,
                                 m_params.ker_w,
                                 m_params.stride_h,
                                 m_params.stride_w,
                                 m_params.pad_h,
                                 m_params.pad_w);
    // store for backward pass
    m_input_cols.emplace_back(im_col);

    auto mean_imcol = im_col.mean(0);
    result = matrix<float>::vstack(result, mean_imcol);
  }

  return Conv2D::reshape_mat2im(result, input.channels(), output_h, output_w);
}

cv::Mat AvgPool2D::backward(const cv::Mat& grad)
{
  // Calculate dimensions
  size_t output_h, output_w;
  std::tie(output_h, output_w) =
      Conv2D::calculate_output_size(m_input.rows, m_input.cols, m_params);

  // Split gradient by channels
  std::vector<cv::Mat> grad_splits;
  cv::split(grad, grad_splits);
  float scale = 1.0f / static_cast<float>(m_params.ker_h * m_params.ker_w);

  std::vector<cv::Mat> dX_per_channels;

  for (size_t c = 0; c < grad_splits.size(); ++c) {
    // Reshape gradient to match the pooling output format
    matrix<float> grad_col = reshape_grad_to_col(grad_splits[c]);

    // Distribute gradient evenly to all elements in each pooling window (with
    // scale)
    matrix<float> dX_col(m_input_cols[c].rows, m_input_cols[c].cols);
    for (size_t i = 0; i < grad_col.cols; ++i) {
      float grad_val = grad_col.data[i] * scale;
      for (size_t j = 0; j < m_params.ker_h * m_params.ker_w; ++j) {
        dX_col.data[j * grad_col.cols + i] = grad_val;
      }
    }

    // Convert back to image format
    cv::Mat grad_channel = Conv2D::col2im(dX_col,
                                          1,  // single channel
                                          m_input.rows,
                                          m_input.cols,
                                          m_params.ker_h,
                                          m_params.ker_w,
                                          m_params.stride_h,
                                          m_params.stride_w,
                                          m_params.pad_h,
                                          m_params.pad_w);

    dX_per_channels.push_back(grad_channel);
  }

  cv::Mat dX;
  cv::merge(dX_per_channels, dX);
  return dX;
}

matrix<float> AvgPool2D::reshape_grad_to_col(const cv::Mat& grad_channel)
{
  matrix<float> col(1, grad_channel.rows * grad_channel.cols);
  size_t i, j;
#pragma omp parallel for private(i, j)
  for (i = 0; i < grad_channel.rows; ++i) {
    for (j = 0; j < grad_channel.cols; ++j) {
      col.data[i * grad_channel.cols + j] = grad_channel.at<float>(i, j);
    }
  }
  return col;
}