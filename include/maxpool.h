#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "layer.h"
#include "matrix.h"

namespace nn
{
class Maxpool : public Layer<float>
{
public:
  Maxpool(size_t size = 2, size_t stride = 2);

  IMPLEMENT_LAYER(float);

private:
  size_t m_size;
  size_t m_stride;
};
};  // namespace nn

#endif  // MAXPOOL_H