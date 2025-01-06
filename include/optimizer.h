#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layers/layer.h"

#define IMPLEMENT_OPTIMIZER() \
  void visit_linear(Linear* layer) override; \
  void visit_conv(Convolution* conv) override;

namespace nn
{
class Linear;
class Convolution;
class Module;

class Optimizer
{
public:
  Optimizer(Module* model, float learning_rate, float weight_decay)
      : m_model(model)
      , m_lr(learning_rate)
      , m_wd(weight_decay) {};

  void step();
  virtual void visit_linear(Linear* layer) = 0;
  virtual void visit_conv(Convolution* conv) = 0;

protected:
  Module* m_model;
  float m_lr;
  float m_wd;
};

class SGD : public Optimizer
{
public:
  SGD(Module* model, float learning_rate, float weight_decay = 0.0f);
  void visit_layer(Layer<float>* layer);
  IMPLEMENT_OPTIMIZER();
};

class AdamW : public Optimizer
{
  // https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
public:
  AdamW(Module* model,
        float learning_rate,
        float weight_decay = 0.0f,
        float m_beta1 = 0.9f,
        float m_beta2 = 0.999f,
        float eps = 1e-8f);
  void visit_layer(Layer<float>* layer);
  IMPLEMENT_OPTIMIZER();

private:
  // (beta_1, beta_2)
  float m_beta1;
  float m_beta2;
  float eps;
  // first movement
  matrix<float> m_0;
  // second movement
  matrix<float> v_0;
};
};  // namespace nn

#endif  // OPTIMIZER_H