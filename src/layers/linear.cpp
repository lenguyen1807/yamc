#include "helper.h"
#include "layers/linear.h"
#include "optimizer.h"

using namespace nn;

Linear::Linear(size_t input_size, size_t output_size, bool rand_init, bool bias)
    : m_b(output_size, 1)
    , m_W(output_size, input_size)
    , is_bias(bias)
{
  if (rand_init) {
    // https://cs231n.github.io/neural-networks-2/#init
    m_W = he_initialize(output_size, input_size);
    ::print_stats(m_W, "weight_initialization");
  }
}

matrix<float> Linear::forward(const matrix<float>& input)
{
  m_input = input;
  if (is_bias)
    return m_W * m_input + m_b;
  else
    return m_W * m_input;
}

matrix<float> Linear::backward(const matrix<float>& grad)
{
  m_dx = m_W.t() * grad;
  m_dW = grad * m_input.t();

  // We only update bias when we need it (bias = True)
  // Else just keep it 0
  if (is_bias) {
    m_db = grad;
  }

  return m_dx;
}

void Linear::zero_grad()
{
  m_dx.fill(0.0f);
  m_dW.fill(0.0f);
  if (is_bias) {
    m_db.fill(0.0f);
  }
}

void Linear::set_weight(const matrix<float>& new_weight)
{
  m_W = new_weight;
}

void Linear::set_bias(const matrix<float>& new_bias)
{
  m_b = new_bias;
}

void Linear::accept_optimizer(Optimizer* optim)
{
  optim->visit_linear(this);
}

void Linear::print_stats()
{
  std::cout << "Linear layer stats:\n";
  ::print_stats(m_W, "weight");
  ::print_stats(m_b, "bias");
  ::print_stats(m_dW, "weight gradient");
  ::print_stats(m_db, "bias gradient");
  ::print_stats(m_dx, "gradient");
}