#include "loss.h"
#include "mlp.h"
#include "utils.h"

nn::CrossEntropyLoss::CrossEntropyLoss(MLP* model)
    : m_pmodel(model)
{
}

double nn::CrossEntropyLoss::operator()(const nn::matrix<double>& logits,
                                        const nn::matrix<double>& label)
{
  assert(logits.cols == 1);
  assert(label.cols == 1);
  assert(logits.rows == label.rows);

  // calculate prediction with softmax
  m_pred = nn::softmax(logits);

  // calculate cross entropy loss
  double entropy = 0.0;
  for (size_t i = 0; i < label.rows; i++) {
    entropy += label.data[i] * std::log(m_pred.data[i]);
  }
  return (-entropy);
}

void nn::CrossEntropyLoss::backward(const nn::matrix<double>& label)
{
  auto loss_grad = m_pred - label;
  m_pmodel->backward(loss_grad);
}

const nn::matrix<double>& nn::CrossEntropyLoss::get_pred() const
{
  return m_pred;
}