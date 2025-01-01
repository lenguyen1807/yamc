#ifndef LOSS_H
#define LOSS_H

#include "activation.h"
#include "matrix.h"

#define IMPLEMENT_LOSS() \
  void backward() override; \
  float operator()(const nn::matrix<float>& logits, \
                   const nn::matrix<float>& label) override;

namespace nn
{
class Module;
class Softmax;

class Loss
{
public:
  Loss(Module* model)
      : m_model(model)
  {
  }

  virtual void backward() = 0;
  virtual float operator()(const matrix<float>& logits,
                           const matrix<float>& label) = 0;

protected:
  Module* m_model;
};

class CrossEntropyLoss : public Loss
{
public:
  CrossEntropyLoss(Module* model);
  IMPLEMENT_LOSS();
  matrix<float> get_pred() const { return m_pred; }

private:
  matrix<float> m_pred;
  matrix<float> m_label;
  Softmax m_activation;
};
};  // namespace nn

#endif  // LOSS_H