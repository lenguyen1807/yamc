#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace nn
{
class MLP;

class Optimizer
{
public:
  virtual void step() {};

protected:
  static auto& get_layers(MLP* model);
};

class SGD : public Optimizer
{
public:
  SGD(MLP* model, float learning_rate);

  void step() override;

private:
  MLP* m_pmodel;
  float m_lr;
};
}  // namespace nn

#endif  // OPTIMIZER_H