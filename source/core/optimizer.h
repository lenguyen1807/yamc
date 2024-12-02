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
  SGD(MLP* model, double learning_rate);

  void step() override;

private:
  MLP* m_pmodel;
  double m_lr;
};
}  // namespace nn

#endif  // OPTIMIZER_H