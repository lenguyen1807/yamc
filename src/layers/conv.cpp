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
    ::print_stats(m_W, "weight_initialization");
  }
}

void Conv2D::accept_optimizer(Optimizer* optim)
{
  optim->visit_conv(this);
}

cv::Mat Conv2D::forward(const cv::Mat& input)
{
  m_im = input.clone();

  m_input = Conv2D::im2col(m_im,
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
  size_t output_h, output_w;
  std::tie(output_h, output_w) =
      Conv2D::calculate_output_size(input.rows, input.cols, m_params);

  return Conv2D::reshape_mat2im(output, output_c, output_h, output_w);
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
  return Conv2D::col2im(dX_col,
                        m_im.channels(),
                        m_im.rows,
                        m_im.cols,
                        m_params.ker_h,
                        m_params.ker_w,
                        m_params.stride_h,
                        m_params.stride_w,
                        m_params.pad_h,
                        m_params.pad_w);
}

std::pair<size_t, size_t> Conv2D::calculate_output_size(size_t input_h,
                                                        size_t input_w,
                                                        ConvParams params)
{
  size_t output_h =
      (input_h + 2 * params.pad_h - params.ker_h) / params.stride_h + 1;
  size_t output_w =
      (input_w + 2 * params.pad_w - params.ker_w) / params.stride_w + 1;
  return std::make_pair(output_h, output_w);
}

matrix<float> Conv2D::im2col(const cv::Mat& data_im,
                             size_t kernel_h,
                             size_t kernel_w,
                             size_t stride_h,
                             size_t stride_w,
                             size_t pad_h,
                             size_t pad_w)
{
  size_t height_col, width_col;
  std::tie(height_col, width_col) = Conv2D::calculate_output_size(
      data_im.rows,
      data_im.cols,
      {kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w});
  size_t channels_col = data_im.channels() * kernel_h * kernel_w;

  matrix<float> res(channels_col, width_col * height_col);

  for (size_t c = 0; c < channels_col; ++c) {
    size_t w_offset = c % kernel_w;
    size_t h_offset = (c / kernel_w) % kernel_h;
    size_t c_im = c / kernel_h / kernel_w;
    for (size_t h = 0; h < height_col; ++h) {
      for (size_t w = 0; w < width_col; ++w) {
        size_t h_im = h * stride_h + h_offset;
        size_t w_im = w * stride_w + w_offset;
        size_t input_idx = (c * height_col + h) * width_col + w;
        res.data[input_idx] =
            Conv2D::get_pixel_im2col(data_im, h_im, w_im, pad_h, pad_w, c_im);
      }
    }
  }

  return res;
}

cv::Mat Conv2D::col2im(const matrix<float> data_im,
                       size_t channels,
                       size_t height,
                       size_t width,
                       size_t kernel_h,
                       size_t kernel_w,
                       size_t stride_h,
                       size_t stride_w,
                       size_t pad_h,
                       size_t pad_w)
{
  // Calculate output dimensions
  size_t height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  size_t width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // Create output image with zeros
  cv::Mat output = cv::Mat::zeros(height, width, CV_32FC(channels));

  // For each channel
  for (size_t c_out = 0; c_out < channels; ++c_out) {
    // For each kernel element
    for (size_t kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
      for (size_t kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
        size_t channel_offset =
            (c_out * kernel_h * kernel_w + kernel_row * kernel_w + kernel_col);

        // For each position in the output
        for (size_t h = 0; h < height_col; ++h) {
          for (size_t w = 0; w < width_col; ++w) {
            size_t h_pad = h * stride_h - pad_h + kernel_row;
            size_t w_pad = w * stride_w - pad_w + kernel_col;

            if (h_pad < height && w_pad < width) {
              // Get value from cols matrix
              size_t col_idx =
                  (channel_offset * height_col + h) * width_col + w;
              float value = data_im.data[col_idx];

              // Add to the appropriate position in output
              if (output.channels() == 1) {
                output.at<float>(h_pad, w_pad) += value;
              } else {
                output.ptr<float>(h_pad, w_pad)[c_out] += value;
              }
            }
          }
        }
      }
    }
  }

  return output;

  // size_t height_col, width_col;
  // std::tie(height_col, width_col) = Conv2D::calculate_output_size(
  //     data_im.rows,
  //     data_im.cols,
  //     {kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w});
  // size_t channels_col = channels * kernel_h * kernel_w;

  // cv::Mat output = cv::Mat::zeros(height, width, CV_32FC(channels));

  // for (size_t c = 0; c < channels_col; ++c) {
  //   size_t w_offset = c % kernel_w;
  //   size_t h_offset = (c / kernel_w) % kernel_h;
  //   size_t c_im = c / kernel_h / kernel_w;
  //   for (size_t h = 0; h < height_col; ++h) {
  //     for (size_t w = 0; w < width_col; ++w) {
  //       size_t h_im = h_offset + h * stride_h;
  //       size_t w_im = w_offset + w * stride_w;
  //       size_t col_index = (c * height_col + h) * width_col + w;
  //       float val = data_im.data[col_index];
  //       Conv2D::add_pixel_col2im(output, h_im, w_im, pad_h, pad_w, c_im,
  //       val);
  //     }
  //   }
  // }

  // return output;
}

float Conv2D::get_pixel_im2col(const cv::Mat& data_im,
                               size_t h_im,
                               size_t w_im,
                               size_t pad_h,
                               size_t pad_w,
                               size_t channel)
{
  // H padding and W padding can be negative
  int h_pad = static_cast<int>(h_im) - static_cast<int>(pad_h);
  int w_pad = static_cast<int>(w_im) - static_cast<int>(pad_w);

  if (h_pad >= 0 && h_pad < data_im.rows && w_pad >= 0 && w_pad < data_im.cols)
  {
    if (data_im.channels() == 1) {
      return data_im.at<float>(h_pad, w_pad);
    } else {
      return data_im.ptr<float>(h_pad, w_pad)[channel];
    }
  }

  return 0.f;
}

void Conv2D::add_pixel_col2im(cv::Mat& data_im,
                              size_t h_im,
                              size_t w_im,
                              size_t pad_h,
                              size_t pad_w,
                              size_t channel,
                              float val)
{
  // H padding and W padding can be negative
  int h_pad = static_cast<int>(h_im) - static_cast<int>(pad_h);
  int w_pad = static_cast<int>(w_im) - static_cast<int>(pad_w);

  if (h_pad >= 0 && h_pad < data_im.rows && w_pad >= 0 && w_pad < data_im.cols)
  {
    if (data_im.channels() == 1) {
      data_im.at<float>(h_pad, w_pad) += val;
    } else {
      data_im.ptr<float>(h_pad, w_pad)[channel] += val;
    }
  }
}

cv::Mat Conv2D::reshape_mat2im(const matrix<float>& im,
                               size_t channels,
                               size_t height,
                               size_t width)
{
  cv::Mat output(height, width, CV_32FC(channels));
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        size_t idx = (c * height + h) * width + w;
        if (channels == 1) {
          output.at<float>(h, w) = im.data[idx];
        } else {
          output.ptr<float>(h, w)[c] = im.data[idx];
        }
      }
    }
  }

  return output;

  // NOTE: This trick is risky af
  // /*
  // Because OpenCV do not support arbitrary channels so we need to use this
  // trick: Create a vector then merge the vector together to create a single
  // image
  // */
  // std::vector<cv::Mat> output_full;

  // for (size_t c = 0; c < channels; c++) {
  //   cv::Mat output_single(height, width, CV_32FC1);

  //   // Find index of each row in 1D vector
  //   /* NOTE: From the first approach, I use a temporary vector to store
  //   std::vector<float> vec = {output.data.begin() ..., ...};
  //   But this is absolutely because res will be destroyed and output_single
  //   will lost data */
  //   float* dst = output_single.ptr<float>(0);
  //   std::copy(im.data.begin() + c * im.cols,
  //             im.data.begin() + (c + 1) * im.cols,
  //             dst);

  //   // Add image to vector
  //   // NOTE: We need to copy opencv image with clone (dont emplace it
  //   directly) output_full.emplace_back(output_single.clone());
  // }

  // cv::Mat output_result(height, width, CV_32FC(channels));
  // cv::merge(output_full, output_result);

  // return output_result;
}

matrix<float> Conv2D::reshape_im2mat(const cv::Mat& im)
{
  matrix<float> res(im.channels(), im.rows * im.cols);

  for (size_t c = 0; c < im.channels(); c++) {
    for (size_t h = 0; h < im.rows; h++) {
      for (size_t w = 0; w < im.cols; w++) {
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