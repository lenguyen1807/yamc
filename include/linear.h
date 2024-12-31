#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include "matrix.h"

namespace nn
{
class Linear : public Layer<float>
{
public:
  Linear(size_t input_size,
         size_t output_size,
         bool rand_init = true,
         bool bias = true);

  IMPLEMENT_LAYER(float);

  void zero_grad() override;
  void set_parameter(const matrix<float>& new_weight,
                     const matrix<float>& new_bias);

private:
  // bias
  matrix<float> m_b;
  // weight
  matrix<float> m_W;
  // weight gradient
  matrix<float> m_dW;
  // input gradient
  matrix<float> m_dx;
  // bias gradient
  matrix<float> m_db;
  // is bias ?
  bool m_bias;
};
};  // namespace nn

#endif  // LINEAR_H