#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <string>

constexpr size_t EPOCHS = 5;

template<typename T>
class matrix;

namespace nn
{

enum class activation
{
  LINEAR,
  SIGMOID,
  RELU,
};

struct layer_config
{
  size_t input;
  size_t output;
  activation activation;
};

namespace F
{

inline double sigmoid(double x, bool grad = false)
{
  if (grad) {
    return sigmoid(x) * (1 - sigmoid(x));
  }

  // stable sigmoid
  // https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
  if (x >= 0.0) {
    return 1.0 / std::exp(-x);
  }
  return std::exp(x) / (1.0 + std::exp(x));
}

inline double relu(double x, bool grad = false)
{
  if (grad) {
    return x >= 0.0 ? 1.0 : 0.0;
  }
  return x >= 0.0 ? x : 0.0;
}

inline double linear(double x, bool grad = false)
{
  if (grad) {
    return 1.0;
  }
  return x;
}

matrix<double> softmax(const matrix<double>& mat);
double crossEntropyLoss(const matrix<double>& pred,
                        const matrix<double>& label);

}  // namespace F
}  // namespace nn

#endif  // UTILS_H