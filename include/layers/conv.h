#ifndef CONV_H
#define CONV_H

#include <cstddef>
#include <utility>

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
  struct Params
  {
    size_t kernel_size;
    size_t padding;
    size_t stride;
  };

  /* NOTE: This Convolution layer supports square image (H = W) only */
  Convolution(size_t input_channels, size_t output_channels, Params params);

  /*
  - Because we are using matrix, we need to convert convolution operation to
  some matrix operations
  - So we use two method called im2col and col2im
  */

  /*
  - You can see an example of im2col here:
  https://ieeexplore.ieee.org/document/9114626/
  - And some implementation of im2col and col2im
  https://github.com/pjreddie/darknet/blob/master/src/im2col.c
  https://github.com/pjreddie/darknet/blob/master/src/col2im.c
  */
  float im2col_get_pixel(const cv::Mat& image,
                         size_t height,
                         size_t width,
                         size_t channels,
                         size_t row,
                         size_t col,
                         size_t channel,
                         size_t pad);

  matrix<float> im2col(const cv::Mat& image,
                       size_t channels,
                       size_t height,
                       size_t width,
                       Params params);

  // calculate output size (h, w)
  static size_t calculate_output_size(size_t input_size, Params params);

  /*
  - Because my matrix isn't a 3d array
  - So I need to use OpenCV to handle image
  */
  matrix<float> forward(const cv::Mat& input);
  void accept_optimizer(Optimizer* optim) override;

private:
  matrix<float> m_W;
  matrix<float> m_b;
  matrix<float> m_inputCol;
  cv::Mat m_inputCache;
};
};  // namespace nn

#endif  // CONV_H