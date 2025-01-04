#include "layers/conv.h"
#include "layers/linear.h"
#include "models/module.h"
#include "optimizer.h"

using namespace nn;

void Optimizer::step()
{
  for (const auto& layer : m_model->layers) {
    layer.second->accept_optimizer(this);
  }
}

SGD::SGD(Module* model, float learning_rate)
    : Optimizer(model)
    , m_lr(learning_rate)
{
}

void SGD::visit_linear(Linear* linear)
{
  auto W = linear->get_weight();
  auto dW = linear->get_weightgrad();
  linear->set_weight(W - (dW % m_lr));

  if (linear->is_bias) {
    auto b = linear->get_bias();
    auto db = linear->get_biasgrad();
    linear->set_bias(b - (db % m_lr));
  }
}

void SGD::visit_conv(Convolution* conv)
{
  // TODO: For later
}