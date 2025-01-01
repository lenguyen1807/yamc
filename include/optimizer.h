#ifndef OPTIMIZER_H
#define OPTIMIZER_H

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
  Optimizer(Module* model)
      : m_model(model) {};

  void step();
  virtual void visit_linear(Linear* layer) = 0;
  virtual void visit_conv(Convolution* conv) = 0;

protected:
  Module* m_model;
};

class SGD : public Optimizer
{
public:
  SGD(Module* model, float learning_rate);
  IMPLEMENT_OPTIMIZER();

private:
  float m_lr;
};
};  // namespace nn

#endif  // OPTIMIZER_H