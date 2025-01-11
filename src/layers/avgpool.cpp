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
  m_input = input.clone();

  // Calculate output size
  size_t output_h, output_w;
  std::tie(output_h, output_w) =
      Conv2D::calculate_output_size(input.rows, input.cols, m_params);

  /* Then we find average value for each columns in each channels for image
   * (through im2col) (seperate by rows). The easiest implementation is
   * splitting image to each channel, then process each channel seperately so I
   * won't care about the index so much */
  std::vector<cv::Mat> input_splits;
  cv::split(m_input, input_splits);
  matrix<float> result;

  for (const auto& img : input_splits) {
    auto im_col = Conv2D::im2col(img,
                                 m_params.ker_h,
                                 m_params.ker_w,
                                 m_params.stride_h,
                                 m_params.stride_w,
                                 m_params.pad_h,
                                 m_params.pad_w);
    auto mean_imcol = im_col.mean(0);
    result = matrix<float>::vstack(result, mean_imcol);
  }

  return Conv2D::reshape_mat2im(result, input.channels(), output_h, output_w);
}

cv::Mat AvgPool2D::backward(const cv::Mat& grad)
{
  // TODO: Implement later
}