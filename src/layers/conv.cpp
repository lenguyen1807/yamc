#include <cstddef>

#include <opencv2/core/mat.hpp>

#include "helper.h"
#include "layers/conv.h"
#include "optimizer.h"

using namespace nn;

Conv2D::Conv2D(size_t input_channels,
               size_t output_channels,
               size_t stride,
               size_t kernel_size,
               size_t padding,
               bool rand_init)
    : Layer<float>(output_channels,
                   input_channels * kernel_size * kernel_size,
                   0,
                   0,
                   false)
    , m_params({kernel_size, kernel_size, padding, padding, stride, stride})
{
  if (rand_init) {
    m_W = xavier_initialize(output_channels,
                            input_channels * kernel_size * kernel_size);
  }
}

void Conv2D::accept_optimizer(Optimizer* optim)
{
  optim->visit_conv(this);
}

cv::Mat Conv2D::forward(const cv::Mat& input)
{
  m_im_channels = input.channels();
  m_im_height = input.rows;
  m_im_width = input.cols;

  m_input = Conv2D::im2col(input,
                           m_params.ker_h,
                           m_params.ker_w,
                           m_params.stride_h,
                           m_params.stride_w,
                           m_params.pad_h,
                           m_params.pad_w);

  // Calculate convolution operation as matrix multiplication
  matrix<float> output = m_W * m_input;

  // Calculate output size
  size_t output_c = output.rows;
  auto output_size =
      Conv2D::calculate_output_size(m_im_height, m_im_width, m_params);

  return Conv2D::reshape_mat2im(output, output_c, output_size.h, output_size.w);
}

cv::Mat Conv2D::backward(const cv::Mat& grad)
{
  /*
   * First reshape the gradient to a matrix
   * Each row vector is an image channel (with height * width dimension)
   */
  matrix<float> grad_mat = Conv2D::reshape_im2mat(grad);

  // Then dX (column version of image gradient) = m_W.T * grad_mat;
  matrix<float> dX_col = m_W.t() * grad_mat;

  // We also need to take gradient w.r.t W
  // We already use im2col on input to get m_input in forward pass
  m_dW = grad_mat * m_input.t();

  // Return image
  cv::Mat output(m_im_height, m_im_width, CV_32FC(m_im_channels));
  Conv2D::col2im(output,
                 dX_col,
                 m_params.ker_h,
                 m_params.ker_w,
                 m_params.stride_h,
                 m_params.stride_w,
                 m_params.pad_h,
                 m_params.pad_w);
  return output.clone();
}

ConvOutputSize Conv2D::calculate_output_size(size_t input_h,
                                             size_t input_w,
                                             ConvParams params)
{
  size_t output_h =
      (input_h + 2 * params.pad_h - params.ker_h) / params.stride_h + 1;
  size_t output_w =
      (input_w + 2 * params.pad_w - params.ker_w) / params.stride_w + 1;
  return {output_h, output_w};
}

matrix<float> Conv2D::im2col(const cv::Mat& data_im,
                             size_t kernel_h,
                             size_t kernel_w,
                             size_t stride_h,
                             size_t stride_w,
                             size_t pad_h,
                             size_t pad_w)
{
  auto output_size = Conv2D::calculate_output_size(
      data_im.rows,
      data_im.cols,
      {kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w});
  size_t channels_col = data_im.channels() * kernel_h * kernel_w;

  matrix<float> res(channels_col, output_size.h * output_size.w);

  for (size_t c = 0; c < channels_col; ++c) {
    size_t w_offset = c % kernel_w;
    size_t h_offset = (c / kernel_w) % kernel_h;
    size_t c_im = c / kernel_h / kernel_w;
    for (size_t h = 0; h < output_size.h; ++h) {
      for (size_t w = 0; w < output_size.w; ++w) {
        size_t h_im = h * stride_h + h_offset;
        size_t w_im = w * stride_w + w_offset;
        size_t input_idx = (c * output_size.h + h) * output_size.w + w;

        // H padding and W padding can be negative
        int h_pad = static_cast<int>(h_im) - static_cast<int>(pad_h);
        int w_pad = static_cast<int>(w_im) - static_cast<int>(pad_w);

        if (h_pad >= 0 && h_pad < data_im.rows && w_pad >= 0
            && w_pad < data_im.cols)
        {
          if (data_im.channels() == 1) {
            res.data[input_idx] = data_im.at<float>(h_pad, w_pad);
          } else {
            res.data[input_idx] = data_im.ptr<float>(h_pad, w_pad)[c_im];
          }
        } else {
          res.data[input_idx] = 0.f;
        }
      }
    }
  }
  return res;
}

void Conv2D::col2im(cv::Mat& im,
                    const matrix<float>& data_im,
                    size_t kernel_h,
                    size_t kernel_w,
                    size_t stride_h,
                    size_t stride_w,
                    size_t pad_h,
                    size_t pad_w)
{
  auto output_size = Conv2D::calculate_output_size(
      im.rows, im.cols, {kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w});
  size_t channels_col = im.channels() * kernel_h * kernel_w;

  for (size_t c = 0; c < channels_col; ++c) {
    size_t w_offset = c % kernel_w;
    size_t h_offset = (c / kernel_w) % kernel_h;
    size_t c_im = c / kernel_h / kernel_w;
    for (size_t h = 0; h < output_size.h; ++h) {
      for (size_t w = 0; w < output_size.w; ++w) {
        size_t h_im = h_offset + h * stride_h;
        size_t w_im = w_offset + w * stride_w;
        size_t col_index = (c * output_size.h + h) * output_size.w + w;
        float val = data_im.data[col_index];

        // H padding and W padding can be negative
        int h_pad = static_cast<int>(h_im) - static_cast<int>(pad_h);
        int w_pad = static_cast<int>(w_im) - static_cast<int>(pad_w);

        if (h_pad >= 0 && h_pad < im.rows && w_pad >= 0 && w_pad < im.cols) {
          if (im.channels() == 1) {
            im.at<float>(h_pad, w_pad) += val;
          } else {
            im.ptr<float>(h_pad, w_pad)[c_im] += val;
          }
        }
      }
    }
  }
}

cv::Mat Conv2D::reshape_mat2im(const matrix<float>& im,
                               size_t channels,
                               size_t height,
                               size_t width)
{
  cv::Mat output(height, width, CV_32FC(channels));
  for (size_t c = 0; c < channels; ++c) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        size_t idx = (c * height + h) * width + w;
        if (channels == 1) {
          output.at<float>(h, w) = im.data[idx];
        } else {
          output.ptr<float>(h, w)[c] = im.data[idx];
        }
      }
    }
  }

  return output.clone();
}

matrix<float> Conv2D::reshape_im2mat(const cv::Mat& im)
{
  matrix<float> res(im.channels(), im.rows * im.cols);

  for (size_t c = 0; c < im.channels(); ++c) {
    for (size_t h = 0; h < im.rows; ++h) {
      for (size_t w = 0; w < im.cols; ++w) {
        size_t idx = c * (im.rows * im.cols) + h * im.cols + w;
        if (im.channels() == 1) {
          res.data[idx] = im.at<float>(h, w);
        } else {
          res.data[idx] = im.ptr<float>(h, w)[c];
        }
      }
    }
  }

  return res;
}

matrix<float> Conv2D::reshape_grad_to_col(const cv::Mat& grad_channel)
{
  matrix<float> col(1, grad_channel.rows * grad_channel.cols);
  for (size_t i = 0; i < grad_channel.rows; ++i) {
    for (size_t j = 0; j < grad_channel.cols; ++j) {
      col.data[i * grad_channel.cols + j] = grad_channel.at<float>(i, j);
    }
  }
  return col;
}