#include <memory>

#include "conv.h"

using namespace nn;

Convolution::Convolution(size_t kernel_size,
                         size_t num_kernels,
                         size_t stride,
                         size_t padding,
                         bool rand_init,
                         bool bias)
    : m_kernelSize(kernel_size)
    , m_numKernels(num_kernels)
    , m_stride(stride)
    , m_padding(padding)
    , m_bias(bias)
    , m_b(num_kernels, 1)
    , m_Ws(num_kernels)
{
  for (size_t kernel = 0; kernel < m_Ws.size(); kernel++) {
    if (rand_init) {
      m_Ws[kernel] = std::make_unique<matrix<float>>(
          matrix<float>::nrand(num_kernels, num_kernels, 0.0f, 1.0f) % 0.01f);
    } else {
      m_Ws[kernel] = std::make_unique<matrix<float>>(num_kernels, num_kernels);
    }
  }
}