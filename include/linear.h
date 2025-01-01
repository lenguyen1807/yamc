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
  void accept_optimizer(Optimizer* optim) override;
  void set_parameter(const matrix<float>& new_weight,
                     const matrix<float>& new_bias);

  matrix<float> get_bias() const { return m_b; }
  matrix<float> get_weight() const { return m_W; }

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