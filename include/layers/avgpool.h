#ifndef AVGPOOL_H
#define AVGPOOL_H

#include <opencv2/core/mat.hpp>

#include "helper.h"
#include "layer.h"

namespace nn
{
class AvgPool2D : public Layer<float>
{
public:
  AvgPool2D(size_t kernel_size, size_t stride, size_t padding = 0);
  IMPLEMENT_LAYER_IM();

private:
  ConvParams m_params;
  std::vector<matrix<float>> m_input_cols;
};
};  // namespace nn

#endif  // AVGPOOL_H