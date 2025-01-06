#ifndef CONV_H
#define CONV_H

#include "layer.h"
#include "matrix.h"

namespace nn
{
template<typename T>
using image = std::vector<matrix<T>>;

class Convolution : public Layer<float>
{
public:
  Convolution(size_t kernel_size,
              size_t num_kernels,
              size_t stride = 1,
              size_t padding = 0,
              bool rand_init = true,
              bool bias = true);

  // We cannot implement these naive forward and backward functions
  // Because an image has three dimension C, H, W instead of only two in matrix
  // IMPLEMENT_LAYER(float);

  // This function will translate convolution operation to matrix multiplication
  // Really powerful
  // matrix<float> im2col(const image<float>& input,
  //                      size_t stride,
  //                      size_t padding);

private:
  size_t m_kernelSize;
  size_t m_numKernels;
  size_t m_stride;
  size_t m_padding;
  bool m_bias;

  matrix<float> m_b;
  std::vector<std::unique_ptr<matrix<float>>> m_Ws;
};
}  // namespace nn

#endif  // CONV_H