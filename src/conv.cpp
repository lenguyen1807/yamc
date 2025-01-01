#include "conv.h"
#include "optimizer.h"

using namespace nn;

Convolution::Convolution(size_t input_size,
                         size_t output_size,
                         size_t kernel_size,
                         size_t stride,
                         size_t padding)
    : m_b(output_size, 1)
{
}

void Convolution::accept_optimizer(Optimizer* optim)
{
  return optim->visit_conv(this);
}

matrix<float> Convolution::im2col(const cv::Mat& image,
                                  size_t kernel_h,
                                  size_t kernel_w,
                                  size_t stride_h,
                                  size_t stride_w,
                                  size_t padding_h,
                                  size_t padding_w)
{
  int channels = image.channels();
  int height = image.rows;
  int width = image.cols;

  int height_col = (height + 2 * padding_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * padding_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  matrix<float> output(channels_col, height_col * width_col);

  cv::Mat padded_image;
  if (padding_h > 0 || padding_w > 0) {
    cv::copyMakeBorder(image,
                       padded_image,
                       padding_h,
                       padding_h,
                       padding_w,
                       padding_w,
                       cv::BORDER_CONSTANT,
                       0);
  } else {
    padded_image = image;
  }

#pragma omp parallel
  {
#pragma omp for collapse(3) schedule(guided)
    for (int c = 0; c < channels; c++) {
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          // Each thread will process this
          for (int h = 0; h < height; h++) {
            int input_row = kh + h * stride_h;
            for (int w = 0; w < width; w++) {
              int input_col = kw + w * stride_w;
              int input_idx =
                  (c * padded_image.rows + input_row) * padded_image.cols
                  + input_col;
              int output_idx =
                  ((c * kernel_h * kernel_w + kh * kernel_w + kw) * height_col
                   + h)
                      * width_col
                  + w;
              output.data[output_idx] = padded_image.at<float>(input_idx);
            }
          }
        }
      }
    }
  }

  return output;
}