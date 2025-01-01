#ifndef CONV_H
#define CONV_H

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "layer.h"
#include "matrix.h"

namespace nn
{
class Optimizer;

class Convolution : Layer<float>
{
public:
  Convolution(size_t input_size,
              size_t output_size,
              size_t kernel_size,
              size_t stride,
              size_t padding);
  void accept_optimizer(Optimizer* optim) override;

  /*
  - Because we are using matrix, we need to convert convolution operation to
  some matrix operations
  - So we use two method called im2col and col2im
  */
  static matrix<float> im2col(const cv::Mat& image,
                              size_t kernel_h,
                              size_t kernel_w,
                              size_t stride_h,
                              size_t stride_w,
                              size_t padding_h,
                              size_t padding_w);
  static cv::Mat col2im(const matrix<float>& col,
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t kernel_h,
                        size_t kernel_w,
                        size_t stride_h,
                        size_t stride_w,
                        size_t padding_h,
                        size_t padding_w);

private:
  matrix<float> m_W;
  matrix<float> m_b;
};
};  // namespace nn

#endif  // CONV_H