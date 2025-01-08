#include "helper.h"
#include "layers/conv.h"
#include "optimizer.h"

using namespace nn;

Convolution::Convolution(size_t input_channels,
                         size_t output_channels,
                         size_t stride,
                         size_t padding,
                         size_t kernel_size,
                         bool rand_init,
                         bool bias)
    : Layer<float>(output_channels,
                   input_channels * kernel_size * kernel_size,
                   output_channels,
                   1,
                   bias)
    , m_params({kernel_size, kernel_size, padding, padding, stride, stride})
{
  if (rand_init) {
    m_W = xavier_initialize(output_channels,
                            input_channels * kernel_size * kernel_size);
  }
}

void Convolution::accept_optimizer(Optimizer* optim)
{
  optim->visit_conv(this);
}

cv::Mat Convolution::forward(const cv::Mat& input)
{
  m_input = input.clone();

  matrix<float> input_cols = im2col(input,
                                    m_params.ker_h,
                                    m_params.ker_w,
                                    m_params.stride_h,
                                    m_params.stride_w,
                                    m_params.pad_h,
                                    m_params.pad_w);

  // Calculate convolution operation as matrix multiplication
  matrix<float> output = m_W * input_cols + m_b;

  // TODO: Implement later
}

cv::Mat Convolution::backward(const cv::Mat& grad)
{
  // TODO: Implement later
}

std::pair<size_t, size_t> Convolution::calculate_output_size(
    size_t input_h, size_t input_w, Convolution::Params params)
{
  size_t output_h =
      (input_h + 2 * params.pad_h - params.ker_h) / params.stride_h + 1;
  size_t output_w =
      (input_w + 2 * params.pad_w - params.ker_w) / params.stride_w + 1;
  return std::make_pair(output_h, output_w);
}

matrix<float> Convolution::im2col(const cv::Mat& data_im,
                                  size_t kernel_h,
                                  size_t kernel_w,
                                  size_t stride_h,
                                  size_t stride_w,
                                  size_t pad_h,
                                  size_t pad_w)
{
  size_t height_col, width_col;
  std::tie(height_col, width_col) = Convolution::calculate_output_size(
      data_im.rows,
      data_im.cols,
      {kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w});
  size_t channels_col = data_im.channels() * kernel_h * kernel_w;

  matrix<float> res(channels_col, width_col * height_col);

#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < channels_col; ++c) {
    for (size_t h = 0; h < height_col; ++h) {
      for (size_t w = 0; w < width_col; ++w) {
        size_t w_offset = c % kernel_w;
        size_t h_offset = (c / kernel_w) % kernel_h;
        size_t c_im = c / (kernel_h * kernel_w);
        size_t h_im = h * stride_h + h_offset;
        size_t w_im = w * stride_w + w_offset;
        size_t input_idx = (c * height_col + h) * width_col + w;
        res.data[input_idx] = Convolution::get_pixel_im2col(
            data_im, h_im, w_im, pad_h, pad_w, c_im);
      }
    }
  }

  return res;
}

float Convolution::get_pixel_im2col(const cv::Mat& data_im,
                                    size_t h_im,
                                    size_t w_im,
                                    size_t pad_h,
                                    size_t pad_w,
                                    size_t channel)
{
  size_t h_pad = h_im - pad_h;
  size_t w_pad = w_im - pad_w;

  if (h_pad >= 0 && h_pad < data_im.cols && w_pad >= 0 && w_pad < data_im.cols)
  {
    if (data_im.channels() == 1) {
      return data_im.at<float>(h_pad, w_pad);
    } else {
      return data_im.at<cv::Vec3f>(h_pad, w_pad)[channel];
    }
  }

  return 0.f;
}