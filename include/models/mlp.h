#ifndef MLP_H
#define MLP_H

#include "matrix.h"
#include "module.h"

class MLP : public nn::Module
{
public:
  // same for __init__ in pytorch
  MLP(size_t input_size, size_t output_size);

  // Now declare forward and backward function
  nn::matrix<float> forward(const nn::matrix<float>& input);
  void backward(const nn::matrix<float>& grad);
};

#endif  // MLP_H