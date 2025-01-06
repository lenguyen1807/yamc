#include "layers/dropout.h"

using namespace nn;

Dropout::Dropout(float p)
    : m_p(p)
{
}

matrix<float> Dropout::forward(const matrix<float>& input)
{
  /* We use a mask, but I'm too lazy so set it as a m_input */
  if (train) {
    // Only calculate when in training
    m_input = matrix<float>::brand(input.rows, input.cols, m_p);
    return input % m_input;
  }
  return input;
}

matrix<float> Dropout::backward(const matrix<float>& grad)
{
  return m_input % grad;
}