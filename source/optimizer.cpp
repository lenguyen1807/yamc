#include "mlp.h"
#include "optimizer.h"

auto& nn::Optimizer::get_layers(MLP* model)
{
  return model->m_layers;
}

nn::SGD::SGD(MLP* model, double learning_rate)
    : m_pmodel(model)
    , m_lr(learning_rate)
{
}

void nn::SGD::step()
{
  for (auto& layer : Optimizer::get_layers(m_pmodel)) {
    auto weight = layer->get_weight();
    auto weight_grad = layer->get_weightgrad();
    layer->set_weight(weight - (weight_grad % m_lr));
  }
}