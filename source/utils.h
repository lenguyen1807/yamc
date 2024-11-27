#ifndef UTILS_H
#define UTILS_H

#include <string>

constexpr size_t EPOCHS = 5;

namespace nn
{

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

namespace F
{
double sigmoid(double x, bool grad = false);
double relu(double x, bool grad = false);
double leakyRelu(double x, double slope, bool grad = false);
double linear(double x, bool grad = false);

template<typename T>
class matrix;

matrix<double> softmax(const matrix<double>& mat);
double crossEntropyLoss(const matrix<double>& pred,
                        const matrix<double>& label);
}  // namespace F
}  // namespace nn

#endif  // UTILS_H