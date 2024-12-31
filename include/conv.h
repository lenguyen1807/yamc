#ifndef CONV_H
#define CONV_H

#include "layer.h"
#include "matrix.h"

namespace nn
{
class Convolution : public Layer<float>
{
public:
  Convolution(size_t kernel_size,
              size_t num_kernels,
              size_t stride = 1,
              size_t padding = 0,
              bool rand_init = true,
              bool bias = true);

  IMPLEMENT_LAYER(float);

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