#include <cmath>

#include "matrix.h"
#include "utils.h"

namespace nn
{
matrix<double> softmax(const matrix<double>& mat)
{
  matrix<double> res(mat);
  double total = 0.0;

  for (size_t i = 0; i < res.rows; i++) {
    total += std::exp(res.data[i]);
  }

  for (size_t i = 0; i < res.rows; i++) {
    res.data[i] = (std::exp(res.data[i]) / total);
  }

  return res;
}

double cross_entropy_loss(const matrix<double>& pred,
                          const matrix<double>& label)
{
  assert(pred.cols == 1);
  assert(label.cols == 1);
  assert(pred.rows == label.rows);

  // calculate cross entropy loss
  double entropy = 0.0;
  for (size_t i = 0; i < label.rows; i++) {
    entropy += label.data[i] * std::log(pred.data[i]);
  }
  return (-entropy);
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

std::string loss_name(nn::Loss loss)
{
  switch (loss) {
    case nn::Loss::CROSS_ENTROPY_LOSS:
      return "Cross Entropy Loss";
  }
}