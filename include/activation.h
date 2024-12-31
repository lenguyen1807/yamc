#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"

namespace nn
{
// An activation function is also a layer
// But it should be an implement of layer
class ReLU : public Layer<float>
{
public:
  IMPLEMENT_LAYER(float);
};

class Softmax : public Layer<float>
{
public:
  IMPLEMENT_LAYER(float);
};
};  // namespace nn

#endif  // ACTIVATION_H