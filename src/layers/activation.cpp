#include "layers/activation.h"

namespace nn
{
matrix<float> ReLU::forward(const matrix<float>& input)
{
  m_input = input;
  return matrix<float>::where(
      m_input > 0.0f, m_input, matrix<float>::values_like(0.0f, m_input));
}

matrix<float> ReLU::backward(const matrix<float>& grad)
{
  return matrix<float>::where(
      m_input > 0.0f, grad, matrix<float>::values_like(0.0f, m_input));
}

matrix<float> Softmax::forward(const matrix<float>& input)
{
  float max_input = matrix<float>::max(input);
  matrix<float> e_x =
      (input - max_input).apply([](float x) { return std::expf(x); });

  // This is a trick because I'm too lazy to implement / a scalar
  return e_x % (1.0f / (e_x.reduce_sum()));
}

matrix<float> Softmax::backward(const matrix<float>& grad)
{
  // You should left softmax derivative for loss function
  return grad;
}
};  // namespace nn