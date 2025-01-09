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
    size_t pad_h;
    size_t pad_w;
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
  - Because my matrix isn't a 3d array
  - So I need to use OpenCV to handle image
  */
  cv::Mat forward(const cv::Mat& input);
  cv::Mat backward(const cv::Mat& grad);
  void accept_optimizer(Optimizer* optim) override;

  /*
  - Because we are using matrix, we need to convert convolution operation to
  some matrix operations
  - So we use two method called im2col and col2im
  - You can see an example of im2col here:
    + https://ieeexplore.ieee.org/document/9114626/
    + https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
  - And some implementation of im2col and col2im
    + https://github.com/pjreddie/darknet/blob/master/src/im2col.c
    + https://github.com/pjreddie/darknet/blob/master/src/col2im.c
    + https://github.com/fmassa/torch-nn/blob/master/ConvLua/im2col.c
    + https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
  */
  static matrix<float> im2col(const cv::Mat& data_im,
                              size_t kernel_h,
                              size_t kernel_w,
                              size_t stride_h,
                              size_t stride_w,
                              size_t pad_h,
                              size_t pad_w);
  static float get_pixel_im2col(const cv::Mat& data_im,
                                size_t h_im,
                                size_t w_im,
                                size_t pad_h,
                                size_t pad_w,
                                size_t channel);
  static std::pair<size_t, size_t> calculate_output_size(size_t input_h,
                                                         size_t input_w,
                                                         Params params);

public:
  // reimplement input from layer
  cv::Mat m_input;
  Params m_params;
};
};  // namespace nn

#endif  // CONV_H