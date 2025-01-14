#include <opencv2/core.hpp>

#include "layers/conv.h"
#include "layers/maxpool.h"

using namespace nn;

Maxpool2D::Maxpool2D(size_t kernel_size, size_t stride, size_t padding)
    : m_params({kernel_size, kernel_size, padding, padding, stride, stride})
{
}

cv::Mat Maxpool2D::forward(const cv::Mat& input)
{
  // We dont need to store whole image
  m_im_height = input.rows;
  m_im_width = input.cols;
  m_im_channels = input.channels();
  m_input_cols.clear();
  m_max_idx_full.clear();

  // Calculate output size
  auto output_size =
      Conv2D::calculate_output_size(m_im_height, m_im_width, m_params);

  /* Then we find max value for each columns in each channels for image
   * (through im2col) (seperate by rows). The easiest implementation is
   * splitting image to each channel, then process each channel seperately so I
   * won't care about the index so much */
  std::vector<cv::Mat> input_splits;
  cv::split(input, input_splits);
  matrix<float> result;

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

    // Find max element and index each column

    /* NOTE: We can't use matrix max by axis function
     * Because We need to store max index for backward pass
     * auto max_imcol = im_col.max(0);
     */

    // Store max value
    matrix<float> max_imcol(1, im_col.cols);

    // Store max index
    std::vector<size_t> max_idx_channel(im_col.cols);

    // Find max value and index
    for (size_t j = 0; j < im_col.cols; j++) {
      // Max is first element of columns
      float max_elm = im_col.data[j];
      size_t max_idx = j;

      for (size_t i = 0; i < im_col.rows; i++) {
        float val = im_col.data[i * im_col.cols + j];
        if (val > max_elm) {
          max_elm = val;
          max_idx = i;
        }
      }

      max_imcol.data[j] = max_elm;
      max_idx_channel[j] = max_idx;
    }

    m_max_idx_full.emplace_back(max_idx_channel);
    result = matrix<float>::vstack(result, max_imcol);
  }

  return Conv2D::reshape_mat2im(
      result, m_im_channels, output_size.h, output_size.w);
}

cv::Mat Maxpool2D::backward(const cv::Mat& grad)
{
  // Split gradient by channels
  std::vector<cv::Mat> grad_splits;
  cv::split(grad, grad_splits);

  cv::Mat dX(m_im_height, m_im_width, CV_32FC(m_im_channels));

  std::vector<cv::Mat> dX_channels;
  cv::split(dX, dX_channels);

  // merge 4 channels each time
  for (size_t c = 0; c < grad_splits.size(); ++c) {
    // Reshape gradient to match the pooling output format
    matrix<float> grad_col = Conv2D::reshape_grad_to_col(grad_splits[c]);

    // Different from Average pooling, we only distribute gradient to max
    // index
    matrix<float> dX_col(m_input_cols[c].rows, m_input_cols[c].cols);
    for (size_t col = 0; col < grad_col.cols; ++col) {
      size_t max_idx = m_max_idx_full[c][col];
      dX_col.data[max_idx * grad_col.cols + col] = grad_col.data[col];
    }

    // Convert back to image format
    Conv2D::col2im(dX_channels[c],
                   dX_col,
                   m_params.ker_h,
                   m_params.ker_w,
                   m_params.stride_h,
                   m_params.stride_w,
                   m_params.pad_h,
                   m_params.pad_w);
  }

  cv::merge(dX_channels, dX);

  return dX.clone();
}