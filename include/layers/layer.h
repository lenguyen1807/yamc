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
  Layer(size_t weight_rows,
        size_t weight_cols,
        size_t bias_rows,
        size_t bias_cols,
        bool bias)
      : m_W(weight_rows, weight_cols)
      , m_b(bias_rows, bias_cols)
      , bias(bias)
  {
  }

  Layer() = default;

  /* Virtual method */
  virtual matrix<T> forward(const matrix<T>& input) { return input; };
  virtual matrix<T> backward(const matrix<T>& grad) { return grad; };
  virtual void print_stats() {}
  virtual void zero_grad() {}
  virtual void accept_optimizer(Optimizer* optim) {}
  virtual ~Layer() {};

  /* Some getter and setter */
  matrix<float> get_weight() const { return m_W; }
  matrix<float> get_weightgrad() const { return m_dW; }
  matrix<float> get_bias() const { return m_b; }
  matrix<float> get_biasgrad() const { return m_db; }
  void set_bias(const matrix<float>& new_bias) { m_b = new_bias; }
  void set_weight(const matrix<float>& new_weight) { m_W = new_weight; }

public:
  bool bias = false;
  bool train = false;

protected:
  matrix<T> m_input;
  // Some layer doesn't need this
  matrix<float> m_W;
  matrix<float> m_b;
  matrix<float> m_dW;
  matrix<float> m_db;
};
}  // namespace nn

#endif  // LAYER_H