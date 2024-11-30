#ifndef MLP_H
#define MLP_H

#include <memory>
#include <vector>

#include "linear.h"

namespace nn
{
class MLP
{
public:
  MLP(const std::vector<LayerConfig>& hidden_configs,
      bool rand_init = true,
      Optimizer optimizer = nn::Optimizer::SGD,
      Loss loss_fn = nn::Loss::CROSS_ENTROPY_LOSS,
      double learning_rate = 0.001);

  auto forward(const matrix<double>& input) -> matrix<double>;
  void backward(const matrix<double>& pred, const matrix<double>& label);
  void optimize();

  void print();
  void zero_grad();

private:
  size_t m_input;
  size_t m_output;
  Optimizer m_optim;
  Loss m_loss;
  std::vector<std::unique_ptr<Linear>> m_layers;
  double m_lr;
};
}  // namespace nn

#endif  // MLP_H