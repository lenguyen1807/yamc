#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include <memory>
#include <string>

#include "utils.h"

template<typename T>
class matrix;

using func_double = std::function<double(double, bool)>;
using dmat_ptr = std::unique_ptr<matrix<double>>;

namespace nn
{
class Linear
{
public:
  Linear(size_t input_size,
         size_t ouput_size,
         Activation activation,
         bool randInit = false);

  void forward(const dmat_ptr& input);

  void backward(const dmat_ptr& prevGrad, const dmat_ptr& prevLayer);

  void zero_grad();

  dmat_ptr pre_activ;
  dmat_ptr post_activ;
  dmat_ptr weight;
  dmat_ptr grad;
  dmat_ptr weight_grad;
  func_double activ_func;
};
}  // namespace nn

#endif  // LAYER_H