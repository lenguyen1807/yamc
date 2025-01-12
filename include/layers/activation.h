#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <opencv2/core.hpp>

#include "layer.h"

namespace nn
{
// An activation function is also a layer
class ReLU : public Layer<float>
{
public:
  IMPLEMENT_LAYER(float);
  IMPLEMENT_LAYER_IM();
};

class Softmax : public Layer<float>
{
public:
  IMPLEMENT_LAYER(float);
};

class LeakyReLU : public Layer<float>
{
public:
  LeakyReLU(float slope = 0.01f)
      : m_slope(slope)
  {
  }

  IMPLEMENT_LAYER(float);

private:
  float m_slope;
};
};  // namespace nn

#endif  // ACTIVATION_H