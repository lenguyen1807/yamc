#include "dropout.h"
#include "module.h"

using namespace nn;

void Module::backward(const matrix<float>& loss_grad)
{
  // It should backward only in training
  if (!m_train) {
    return;
  }

  matrix<float> grad(loss_grad);

  // Then update gradients in each layer
  for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter) {
    grad = (*iter)->backward(grad);
  }
}

void Module::zero_grad()
{
  for (const auto& layer : m_layers) {
    layer->zero_grad();
  }
}

matrix<float> Module::forward(const matrix<float>& input)
{
  matrix<float> res(input);
  for (const auto& layer : m_layers) {
    // This is call RTTI (run time type informatoin, a way to check type in
    // run-time)
    if (auto ptr = dynamic_cast<Dropout*>(layer.get())) {
      res = ptr->forward(res, m_train);
    } else {
      res = layer->forward(res);
    }
  }
  return res;
}

void Module::train()
{
  m_train = true;
}

void Module::eval()
{
  m_train = false;
}