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

SGD::SGD(Module* model, float learning_rate, float weight_decay)
    : Optimizer(model, learning_rate, weight_decay)
{
}

void SGD::visit_layer(Layer<float>* layer)
{
  auto W = layer->get_weight();
  auto dW = layer->get_weightgrad();
  layer->set_weight(W % (1.0f - m_lr * m_wd) - (dW % m_lr));

  if (layer->bias) {
    auto b = layer->get_bias();
    auto db = layer->get_biasgrad();
    layer->set_bias(b % (1.0f - m_lr * m_wd) - (db % m_lr));
  }
}

void SGD::visit_linear(Linear* linear)
{
  visit_layer(linear);
}

void SGD::visit_conv(Conv2D* conv)
{
  // It should be the same as linear
  visit_layer(conv);
}

AdamW::AdamW(Module* model,
             float learning_rate,
             float weight_decay,
             float m_beta1,
             float m_beta2,
             float eps)
    : Optimizer(model, learning_rate, weight_decay)
{
}

void AdamW::visit_layer(Layer<float>* layer)
{
  auto W = layer->get_weight();
  auto dW = layer->get_weightgrad();

  // TODO: Implement later

  // update the same for bias
  if (layer->bias) {
    auto b = layer->get_bias();
    auto db = layer->get_biasgrad();
  }
}

void AdamW::visit_conv(Conv2D* conv)
{
  visit_layer(conv);
}

void AdamW::visit_linear(Linear* linear)
{
  visit_layer(linear);
}
