#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <functional>
#include <string>

constexpr size_t EPOCHS = 1;
constexpr size_t SEED = 1234;

namespace nn
{

// forward declaration
template<typename T>
class matrix;

enum class Activation
{
  LINEAR,
  SIGMOID,
  RELU,
};

struct LayerConfig
{
  size_t input;
  size_t output;
  Activation activation;
};

inline float sigmoid(float x, bool grad = false)
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

inline float relu(float x, bool grad = false)
{
  if (grad) {
    return x >= 0.0 ? 1.0 : 0.0;
  }
  return x >= 0.0 ? x : 0.0;
}

inline float linear(float x, bool grad = false)
{
  if (grad) {
    return 1.0;
  }
  return x;
}

matrix<float> softmax(const matrix<float>& mat);

}  // namespace nn

// helper function get activation name
std::string activation_name(nn::Activation activ);

#endif  // UTILS_H