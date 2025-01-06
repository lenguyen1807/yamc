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

private:
  // input gradient
  matrix<float> m_dx;
};
};  // namespace nn

#endif  // LINEAR_H