#include "layers/conv.h"
#include "optimizer.h"

using namespace nn;

Convolution::Convolution(size_t input_channels,
                         size_t output_channels,
                         Params params)
    : m_b(output_channels, 1)
{
}

void Convolution::accept_optimizer(Optimizer* optim)
{
  return optim->visit_conv(this);
}

size_t Convolution::calculate_output_size(size_t input_size,
                                          Convolution::Params params)
{
  // https://arxiv.org/abs/1603.07285
  // page 15, relationship 6
  return (input_size + 2 * params.padding - params.kernel_size) / params.stride
      + 1;
}

matrix<float> Convolution::im2col(const cv::Mat& image,
                                  size_t channels,
                                  size_t height,
                                  size_t width,
                                  Params params)
{
  int height_col = calculate_output_size(height, params);
  int width_col = calculate_output_size(width, params);
  int channels_col = channels * params.kernel_size * params.kernel_size;

  matrix<float> imcols;

  for (int c = 0; c < channels; c++) {
    int w_offset = c % params.kernel_size;
    int h_offset = (c / params.kernel_size) % params.kernel_size;
    int c_im = c / (params.kernel_size * params.kernel_size);
#pragma omp parallel for collapse(2)
    for (int h = 0; h < height_col; h++) {
      for (int w = 0; w < width_col; w++) {
        int im_row = h_offset + h * params.stride;
        int im_col = w_offset + w * params.stride;
        int col_index = (c * height_col + h) * width_col + w;
        imcols.data[col_index] = im2col_get_pixel(image,
                                                  height,
                                                  width,
                                                  channels,
                                                  im_row,
                                                  im_col,
                                                  c_im,
                                                  params.padding);
      }
    }
  }
  return imcols;
}

float Convolution::im2col_get_pixel(const cv::Mat& image,
                                    size_t height,
                                    size_t width,
                                    size_t channels,
                                    size_t row,
                                    size_t col,
                                    size_t channel,
                                    size_t pad)
{
  row -= pad;
  col -= pad;
  if (row < 0 || col < 0 || row >= height || col >= width)
    return 0;
  return image.at<float>(col + width * (row + height * channel));
}