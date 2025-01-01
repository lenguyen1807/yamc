#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

#define IMPLEMENT_LAYER(T) \
  matrix<T> forward(const matrix<T>& input) override; \
  matrix<T> backward(const matrix<T>& grad) override;

namespace nn
{
// forward declaration
class Optimizer;

template<typename T>
class Layer
{
public:
  virtual matrix<T> forward(const matrix<T>& input) {};
  virtual matrix<T> backward(const matrix<T>& grad) {};
  virtual void zero_grad() {}
  virtual void accept_optimizer(Optimizer* optim) {}
  virtual ~Layer() {}

protected:
  matrix<T> m_input;
};
}  // namespace nn

#endif  // LAYER_H