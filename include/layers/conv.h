#ifndef CONV_H
#define CONV_H

#include <cstddef>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "helper.h"
#include "layer.h"
#include "matrix.h"

namespace nn
{
class Optimizer;

class Conv2D : public Layer<float>
{
public:
  Conv2D(size_t input_channels,
         size_t output_channels,
         size_t stride,
         size_t kernel_size,
         size_t padding = 0,
         bool rand_init = true);

  /*
  - Because my matrix isn't a 3d array
  - So I need to use OpenCV to handle image
  */
  IMPLEMENT_LAYER_IM();
  void accept_optimizer(Optimizer* optim) override;

  /*
  - Because we are using matrix, we need to convert convolution operation to
  some matrix operations
  - So we use two method called im2col and col2im
  - You can see an example of im2col here:
    + https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
  - And some implementation of im2col and col2im
    + https://github.com/pjreddie/darknet/blob/master/src/im2col.c
    + https://github.com/pjreddie/darknet/blob/master/src/col2im.c
    + https://github.com/fmassa/torch-nn/blob/master/ConvLua/im2col.c
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
  static cv::Mat col2im(const matrix<float> data_im,
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t kernel_h,
                        size_t kernel_w,
                        size_t stride_h,
                        size_t stride_w,
                        size_t pad_h,
                        size_t pad_w);
  static void add_pixel_col2im(cv::Mat& data_im,
                               size_t h_im,
                               size_t w_im,
                               size_t pad_h,
                               size_t pad_w,
                               size_t channel,
                               float val);
  static std::pair<size_t, size_t> calculate_output_size(size_t input_h,
                                                         size_t input_w,
                                                         ConvParams params);
  static cv::Mat reshape_mat2im(const matrix<float>& im,
                                size_t channels,
                                size_t height,
                                size_t width);
  static matrix<float> reshape_im2mat(const cv::Mat& im);
  static matrix<float> reshape_grad_to_col(const cv::Mat& grad_channel);

public:
  ConvParams m_params;
};
};  // namespace nn

#endif  // CONV_H