#ifndef LINEAR_H
#define LINEAR_H

#include <functional>

#include "matrix.h"
#include "utils.h"

namespace nn
{

class Linear
{
public:
  Linear(size_t input_size,
         size_t output_size,
         Activation activation,
         bool rand_init = false);

  void print();
  void compute(const matrix<double>& input);
  void grad(const matrix<double>& after_grad, const matrix<double>& prev_layer);
  void zero_grad();

  const matrix<double>& get_output() const;
  const matrix<double>& get_preactiv() const;
  const matrix<double>& get_grad() const;
  const matrix<double>& get_weight() const;
  const matrix<double>& get_weightgrad() const;

  void set_weight(const matrix<double>& new_weight);

private:
  matrix<double> m_preactiv;
  matrix<double> m_afteractiv;
  matrix<double> m_weight;
  matrix<double> m_weightgrad;
  matrix<double> m_grad;

  Activation m_activ;
  std::function<double(double, bool)> m_activfunc;
};
}  // namespace nn

#endif  // LINEAR_H