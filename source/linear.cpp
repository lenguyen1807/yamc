#include <cmath>
#include <iostream>

#include "linear.h"
#include "utils.h"

nn::Linear::Linear(size_t input_size,
                   size_t output_size,
                   Activation activation,
                   bool rand_init)
    : m_preactiv(output_size, 1)
    , m_afteractiv(output_size, 1)
    , m_weight(output_size, input_size)
    , m_weightgrad(output_size, input_size)
    , m_grad(input_size, 1)
    , m_activ(activation)
{
  if (rand_init) {
    // https://cs231n.github.io/neural-networks-2/#init
    this->m_weight = matrix<double>::nrand(
        output_size, input_size, 0.0, 2.0 / std::sqrt(input_size));
  }

  switch (m_activ) {
    case nn::Activation::LINEAR:
      m_activfunc = nn::linear;
      break;
    case nn::Activation::RELU:
      m_activfunc = nn::relu;
      break;
    case nn::Activation::SIGMOID:
      m_activfunc = nn::sigmoid;
      break;
  }
}

void nn::Linear::compute(const matrix<double>& input)
{
  m_preactiv = m_weight * input;
  m_afteractiv = m_preactiv.apply(m_activfunc, false);
}

void nn::Linear::grad(const matrix<double>& prev_grad,
                      const matrix<double>& prev_layer)
{
  // step 1: compute gradient for after-activation layer
  matrix<double> after_grad = prev_grad % m_preactiv.apply(m_activfunc, true);

  // step 2: compute gradient respect to weight
  m_weightgrad = after_grad * prev_layer.t();

  // step 3: compute gradient for pre-activation layer
  m_grad = m_weight.t() * after_grad;
}

void nn::Linear::zero_grad()
{
  m_weightgrad.fill(0.0);
  m_grad.fill(0.0);
}

nn::matrix<double>& nn::Linear::get_output()
{
  return m_afteractiv;
}

nn::matrix<double>& nn::Linear::get_grad()
{
  return m_grad;
}

nn::matrix<double>& nn::Linear::get_weight()
{
  return m_weight;
}

nn::matrix<double>& nn::Linear::get_weightgrad()
{
  return m_weightgrad;
}

nn::matrix<double>& nn::Linear::get_preactiv()
{
  return m_preactiv;
}

void nn::Linear::print()
{
  std::cout << "Linear(output_size=" << m_weight.rows
            << ", input_size=" << m_weight.cols
            << ", activation=" << activation_name(m_activ) << ")\n";
}