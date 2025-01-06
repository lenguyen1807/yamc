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

class Convolution : public Layer<float>
{
public:
  struct Params
  {
    size_t ker_h;
    size_t ker_w;
    size_t pad_w;
    size_t pad_h;
    size_t stride_h;
    size_t stride_w;
  };

  Convolution(size_t input_channels,
              size_t output_channels,
              size_t stride,
              size_t padding,
              size_t kernel_size,
              bool rand_init = true,
              bool bias = true);

  /*
  - Because we are using matrix, we need to convert convolution operation to
  some matrix operations
  - So we use two method called im2col and col2im
  - You can see an example of im2col here:
  https://ieeexplore.ieee.org/document/9114626/
  - And some implementation of im2col and col2im
  https://github.com/pjreddie/darknet/blob/master/src/im2col.c
  https://github.com/pjreddie/darknet/blob/master/src/col2im.c
  */
  matrix<float> im2col(const cv::Mat& data_im,
                       size_t channels,
                       size_t height,
                       size_t width,
                       size_t ksize,
                       size_t stride,
                       size_t pad);
  size_t im2col_pixel_index(size_t height,
                            size_t width,
                            size_t channels,
                            size_t row,
                            size_t col,
                            size_t channel,
                            size_t pad);

  /* Some helper function */
  std::pair<size_t, size_t> calculate_output_size(size_t input_h,
                                                  size_t input_w,
                                                  Params params);

  /*
  - Because my matrix isn't a 3d array
  - So I need to use OpenCV to handle image
  */
  matrix<float> forward(const cv::Mat& input);
  void accept_optimizer(Optimizer* optim) override;

public:
  Params params;
};
};  // namespace nn

#endif  // CONV_H