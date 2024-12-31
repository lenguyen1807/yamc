#include "linear.h"

using namespace nn;

Linear::Linear(size_t input_size, size_t output_size, bool rand_init, bool bias)
    : m_b(output_size, 1)
    , m_W(output_size, input_size)
    , m_dx(input_size, 1)
    , m_db(output_size, 1)
    , m_bias(bias)
{
  if (rand_init) {
    // https://cs231n.github.io/neural-networks-2/#init
    m_W = matrix<float>::nrand(
        output_size, input_size, 0.0, 2.0 / static_cast<float>(input_size));
  }
}

matrix<float> Linear::forward(const matrix<float>& input)
{
  m_input = input;
  return m_W * m_input + m_b;
}

matrix<float> Linear::backward(const matrix<float>& grad)
{
  m_dx = m_W.t() * grad;
  m_dW = grad * m_input.t();

  // We only update bias when we need it (bias = True)
  // Else just keep it 0
  if (m_bias) {
    m_db = grad;
  }

  return m_dx;
}

void Linear::zero_grad()
{
  m_dx.fill(0.0f);
  m_dW.fill(0.0f);
  m_db.fill(0.0f);
}

void Linear::set_parameter(const matrix<float>& new_weight,
                           const matrix<float>& new_bias)
{
  m_W = new_weight;
  m_b = new_bias;
}