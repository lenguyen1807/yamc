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
  void compute(const matrix<float>& input);
  void grad(const matrix<float>& after_grad, const matrix<float>& prev_layer);
  void zero_grad();

  const matrix<float>& get_output() const;
  const matrix<float>& get_preactiv() const;
  const matrix<float>& get_grad() const;
  const matrix<float>& get_weight() const;
  const matrix<float>& get_weightgrad() const;

  void set_weight(const matrix<float>& new_weight);

private:
  matrix<float> m_preactiv;
  matrix<float> m_afteractiv;
  matrix<float> m_weight;
  matrix<float> m_weightgrad;
  matrix<float> m_grad;

  Activation m_activ;
  std::function<float(float, bool)> m_activfunc;
};
}  // namespace nn

#endif  // LINEAR_H