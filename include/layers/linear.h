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
  void print_stats() override;
  void set_weight(const matrix<float>& new_weight);
  void set_bias(const matrix<float>& bias);

  matrix<float> get_bias() const { return m_b; }
  matrix<float> get_weight() const { return m_W; }
  matrix<float> get_weightgrad() const { return m_dW; }
  matrix<float> get_biasgrad() const { return m_db; }

public:
  bool is_bias;

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
};
};  // namespace nn

#endif  // LINEAR_H