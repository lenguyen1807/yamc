#include "conv.h"
#include "linear.h"
#include "module.h"
#include "optimizer.h"

using namespace nn;

void Optimizer::step()
{
  for (const auto& layer : m_model->m_layers) {
    layer->accept_optimizer(this);
  }
}

SGD::SGD(Module* model, float learning_rate)
    : Optimizer(model)
    , m_lr(learning_rate)
{
}

void SGD::visit_linear(Linear* linear)
{
  auto weight = linear->get_weight();
  auto bias = linear->get_bias();
  linear->set_parameter(weight - (weight % m_lr), bias - (bias % m_lr));
}

void SGD::visit_conv(Convolution* conv)
{
  // TODO: For later
}