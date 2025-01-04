#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"

namespace nn
{
class Dropout : public Layer<float>
{
public:
  Dropout(float p = .5f);

  IMPLEMENT_LAYER(float);

  // Dropout only "forward" in train mode
  matrix<float> forward(const matrix<float>& input, bool train);

private:
  float m_p;
};
};  // namespace nn

#endif  // DROPOUT_H