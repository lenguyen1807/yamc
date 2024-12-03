#include <cmath>

#include "matrix.h"
#include "utils.h"

namespace nn
{
matrix<float> softmax(const matrix<float>& mat)
{
  matrix<float> res(mat);
  float total = 0.0;

  for (size_t i = 0; i < res.rows; i++) {
    total += std::exp(res.data[i]);
  }

  for (size_t i = 0; i < res.rows; i++) {
    res.data[i] = (std::exp(res.data[i]) / total);
  }

  return res;
}

}  // namespace nn

std::string activation_name(nn::Activation activ)
{
  switch (activ) {
    case nn::Activation::LINEAR:
      return "Linear";
    case nn::Activation::RELU:
      return "ReLU";
    case nn::Activation::SIGMOID:
      return "Sigmoid";
  }
}