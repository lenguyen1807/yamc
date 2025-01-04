#include "layers/activation.h"
#include "loss.h"
#include "models/module.h"

using namespace nn;

CrossEntropyLoss::CrossEntropyLoss(Module* model)
    : Loss(model)
    , m_activation()
{
}

float CrossEntropyLoss::operator()(const matrix<float>& logits,
                                   const matrix<float>& label)
{
  assert(logits.cols == 1);
  assert(label.cols == 1);
  assert(logits.rows == label.rows);

  // Already implement softmax in cross entropy loss
  m_pred = m_activation.forward(logits);

  // cache label for backward
  m_label = label;

  // calculate cross entropy loss
  float entropy = 0.f;
  for (size_t i = 0; i < label.rows; i++) {
    entropy += (label.data[i] * std::logf(m_pred.data[i]));
  }
  return (-entropy);
}

matrix<float> CrossEntropyLoss::get_loss_grad() const
{
  return m_pred - m_label;
}