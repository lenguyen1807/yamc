#include <memory>

#include "linear.h"
#include "mlp.h"
#include "utils.h"

nn::MLP::MLP(const std::vector<nn::LayerConfig>& hidden_configs, bool rand_init)
    : m_input(hidden_configs.front().input)
    , m_output(hidden_configs.back().output)
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
  auto output = m_layers[0]->get_output();

  for (size_t i = 1; i < m_layers.size(); i++) {
    m_layers[i]->compute(output);
    output = m_layers[i]->get_output();
  }

  return output;
}

void nn::MLP::backward(const matrix<double>& loss_grad)
{
  size_t last_idx = m_layers.size() - 1;

  // calculate gradient of the output
  m_layers[last_idx]->grad(loss_grad, m_layers[last_idx - 1]->get_output());
  auto grad = m_layers[last_idx]->get_grad();

  // calculate gradient of other layers
  for (size_t i = last_idx; i-- > 1;) {
    m_layers[i]->grad(grad, m_layers[i - 1]->get_output());
    grad = m_layers[i]->get_grad();
  }
}

void nn::MLP::print()
{
  for (const auto& layer : m_layers) {
    layer->print();
  }
}

void nn::MLP::zero_grad()
{
  for (auto& layer : m_layers) {
    layer->zero_grad();
  }
}