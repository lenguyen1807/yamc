#include "activation.h"
#include "loss.h"
#include "module.h"

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
  float entropy = 0.0;
#pragma omp parallel for
  for (size_t i = 0; i < label.rows; i++) {
    entropy += label.data[i] * std::log(m_pred.data[i]);
  }
  return (-entropy);
}

void CrossEntropyLoss::backward()
{
  m_model->backward(m_pred - m_label);
}