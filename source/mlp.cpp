#include <iostream>
#include <memory>

#include "linear.h"
#include "mlp.h"
#include "utils.h"

nn::MLP::MLP(const std::vector<nn::LayerConfig>& hidden_configs,
             bool rand_init,
             nn::Optimizer optimizer,
             nn::Loss loss_fn,
             double learning_rate)
    : m_input(hidden_configs.front().input)
    , m_output(hidden_configs.back().output)
    , m_optim(optimizer)
    , m_loss(loss_fn)
    , m_lr(learning_rate)
{
  m_layers.reserve(hidden_configs.size());
  for (const auto& config : hidden_configs) {
    m_layers.emplace_back(std::make_unique<nn::Linear>(
        config.input, config.output, config.activation, rand_init));
  }
}

auto nn::MLP::forward(const nn::matrix<double>& input) -> nn::matrix<double>
{
  // calculate input with first hidden layer
  m_layers[0]->compute(input);

  matrix<double> output = m_layers[0]->get_output();

  for (size_t i = 1; i < m_layers.size(); i++) {
    m_layers[i]->compute(output);
    output = m_layers[i]->get_output();
  }

  return output;
}

void nn::MLP::backward(const nn::matrix<double>& pred,
                       const nn::matrix<double>& label)
{
  // calculate gradient with of loss
  matrix<double> loss_grad;
  switch (m_loss) {
    case nn::Loss::CROSS_ENTROPY_LOSS:
      loss_grad = pred - label;
      break;
  }

  size_t last_idx = m_layers.size() - 1;

  // calculate gradient of the output
  m_layers[last_idx]->grad(loss_grad, m_layers[last_idx - 1]->get_output());
  matrix<double> grad = m_layers[last_idx]->get_grad();

  // calculate gradient of other layers
  for (size_t i = last_idx - 1; i > 1; i--) {
    m_layers[i]->grad(grad, m_layers[i - 1]->get_output());
    grad = m_layers[i]->get_grad();
  }
}

void nn::MLP::optimize()
{
  if (m_optim == nn::Optimizer::SGD) {
    int idx = 0;
    for (auto& layer : m_layers) {
      auto weight = layer->get_weight();
      auto weight_grad = layer->get_weightgrad();

      // if (idx == m_layers.size() - 1) {
      //   std::cout << "Weight: \n";
      //   weight.print();
      //   std::cout << "Weight grad: \n";
      //   weight_grad.print();
      // }

      layer->set_weight(weight - (weight_grad % m_lr));

      // if (idx == m_layers.size() - 1) {
      //   std::cout << "New weight: \n";
      //   layer->get_weight().print();
      // }

      idx++;
    }
  }
}

void nn::MLP::print()
{
  for (const auto& layer : m_layers) {
    layer->print();
  }
  std::cout << "Learning rate: " << m_lr << "\n";
  std::cout << "Optimizer: " << optimizer_name(m_optim) << "\n";
  std::cout << "Loss: " << loss_name(m_loss) << "\n";
}

void nn::MLP::zero_grad()
{
  for (auto& layer : m_layers) {
    layer->zero_grad();
  }
}