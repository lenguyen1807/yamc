#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <vector>

#include "helper.h"
#include "layer.h"

namespace nn
{
class Maxpool2D : public Layer<float>
{
public:
  // Actually, I don't want bias for max pooling layer
  Maxpool2D(size_t kernel_size, size_t stride, size_t padding = 0);
  IMPLEMENT_LAYER_IM();

private:
  ConvParams m_params;
  std::vector<matrix<float>> m_input_cols;
  std::vector<std::vector<size_t>> m_max_idx_full;
};
};  // namespace nn

#endif  // MAXPOOL_H