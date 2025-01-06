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
                                    input.channels(),
                                    input.rows,
                                    input.cols,
                                    m_params.ker_h,
                                    m_params.stride_h,
                                    m_params.pad_h);

  // Calculate convolution operation as matrix multiplication
  matrix<float> output = m_W * input_cols;
  // Add bias
  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output.cols; ++j) {
      output.data[i * output.cols + j] += m_b.data[i];
    }
  }

  size_t output_h, output_w;
  std::tie(output_h, output_w) =
      calculate_output_size(input_cols.rows, input_cols.cols, m_params);

  cv::Mat result(output_h, output_w, CV_32F);

  // reshape our matrix to multiple to image
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output.cols; ++j) {
      size_t out_c = i;  // Since each row represents one output channel
      size_t out_h = j / output_w;
      size_t out_w = j % output_w;
      if (input.channels() == 1) {
        result.at<float>(out_h, out_w) = output.data[i * output.cols + j];
      } else {
        result.at<cv::Vec3f>(out_h, out_w)[out_c] =
            output.data[i * output.cols + j];
      }
    }
  }

  return result;
}

cv::Mat Convolution::backward(const cv::Mat& grad)
{
  // TODO: Implement later
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

matrix<float> Convolution::im2col(const cv::Mat& data_im,
                                  size_t channels,
                                  size_t height,
                                  size_t width,
                                  size_t ksize,
                                  size_t stride,
                                  size_t pad)
{
  size_t height_col, width_col;
  std::tie(height_col, width_col) = Convolution::calculate_output_size(
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

        size_t col_index = (c * height_col * width_col) + (h * width_col) + w;
        res.data[col_index] = get_pixel_value(
            data_im, height, width, channels, im_row, im_col, c_im, pad);
      }
    }
  }

  return res;
}

cv::Mat Convolution::col2im(const matrix<float>& col,
                            size_t channels,
                            size_t height,
                            size_t width,
                            size_t ksize,
                            size_t stride,
                            size_t pad)
{
  size_t height_col, width_col;
  std::tie(height_col, width_col) = Convolution::calculate_output_size(
      height, width, {ksize, ksize, pad, pad, stride, stride});
  size_t channels_col = channels * ksize * ksize;

  cv::Mat result = cv::Mat::zeros(height, width, CV_32FC(channels));
#pragma omp parallel for collapse(3)
  for (size_t c = 0; c < channels_col; c++) {
    for (size_t h = 0; h < height_col; ++h) {
      for (size_t w = 0; w < width_col; ++w) {
        size_t w_offset = c % ksize;
        size_t h_offset = (c / ksize) % ksize;
        size_t c_im = c / (ksize * ksize);

        size_t im_row = h_offset + h * stride;
        size_t im_col = w_offset + w * stride;

        size_t col_index = (c * height_col * width_col) + (h * width_col) + w;
        float val = col.data[col_index];
        set_pixel_value(
            result, height, width, channels, im_row, im_col, c_im, pad, val);
      }
    }
  }
  return result;
}

float Convolution::get_pixel_value(const cv::Mat& im,
                                   int height,
                                   int width,
                                   int channels,
                                   int row,
                                   int col,
                                   int channel,
                                   int pad)
{
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width)
    return 0.f;

  if (channels == 1) {
    return im.at<float>(row, col);
  } else {
    return im.at<cv::Vec3f>(row, col)[channel];
  }
}

void Convolution::set_pixel_value(cv::Mat& im,
                                  int height,
                                  int width,
                                  int channels,
                                  int row,
                                  int col,
                                  int channel,
                                  int pad,
                                  float val)
{
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width)
    return;

  if (channels == 1) {
    im.at<float>(row, col) += val;
  } else {
    im.at<cv::Vec3f>(row, col)[channel] += val;
  }
}
