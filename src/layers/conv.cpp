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
{
  if (rand_init) {
    m_W = xavier_initialize(output_channels,
                            input_channels * kernel_size * kernel_size);
  }
}

std::pair<size_t, size_t> Convolution::calculate_output_size(
    size_t input_h, size_t input_w, Convolution::Params params)
{
  size_t output_h =
      (input_h - 2 * params.pad_h + params.ker_h) / params.stride_h + 1;
  size_t output_w =
      (input_w - 2 * params.pad_w + params.ker_w) / params.stride_w + 1;
  return std::make_pair(output_h, output_w);
}

void Convolution::accept_optimizer(Optimizer* optim)
{
  optim->visit_conv(this);
}

matrix<float> Convolution::im2col(const cv::Mat& data_im,
                                  size_t channels,
                                  size_t height,
                                  size_t width,
                                  size_t ksize,
                                  size_t stride,
                                  size_t pad)
{
  size_t height_col, width_col;
  std::tie(height_col, width_col) = calculate_output_size(
      height, width, {ksize, ksize, pad, pad, stride, stride});
  size_t channels_col = channels * ksize * ksize;

  matrix<float> res(channels_col * height_col, width_col);

#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < channels_col; c++) {
    for (size_t h = 0; h < height_col; ++h) {
      for (size_t w = 0; w < width_col; ++w) {
        size_t w_offset = c % ksize;
        size_t h_offset = (c / ksize) % ksize;
        size_t c_im = c / (ksize * ksize);

        size_t im_row = h_offset + h * stride;
        size_t im_col = w_offset + w * stride;

        size_t col_index = (c * height_col + h) * width_col + w;

        size_t output_index = im2col_pixel_index(
            height, width, channels, im_row, im_col, c_im, pad);
        res.data[col_index] = data_im.at<float>(output_index);
      }
    }
  }

  return res;
}

size_t Convolution::im2col_pixel_index(size_t height,
                                       size_t width,
                                       size_t channels,
                                       size_t row,
                                       size_t col,
                                       size_t channel,
                                       size_t pad)
{
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width) {
    return 0;
  }

  // Convert 3D coordinates to 1D index in row-major order
  return col + width * (row + height * channel);
}