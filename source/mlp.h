#ifndef MLP_H
#define MLP_H

#include <memory>
#include <vector>

#include "linear.h"

namespace nn
{

class Optimizer;

class MLP
{
public:
  friend class nn::Optimizer;

  explicit MLP(const std::vector<LayerConfig>& hidden_configs,
               bool rand_init = true,
               Loss loss_fn = nn::Loss::CROSS_ENTROPY_LOSS);

  auto forward(const matrix<double>& input) -> matrix<double>;
  void backward(const matrix<double>& pred, const matrix<double>& label);

  void print();
  void zero_grad();

private:
  size_t m_input;
  size_t m_output;
  Loss m_loss;
  std::vector<std::unique_ptr<Linear>> m_layers;
};
}  // namespace nn

#endif  // MLP_H