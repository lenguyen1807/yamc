#include <cmath>

#include "matrix.h"
#include "utils.h"

using namespace nn::F;

double sigmoid(double x, bool grad)
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

double relu(double x, bool grad)
{
  if (grad) {
    return x >= 0.0 ? 1.0 : 0.0;
  }
  return x >= 0.0 ? x : 0.0;
}

double leakyRelu(double x, double slope, bool grad)
{
  if (grad) {
    return x >= 0.0 ? 1.0 : slope;
  }
  return x >= 0.0 ? x : x * slope;
}

double linear(double x, bool grad)
{
  if (grad) {
    return 1.0;
  }
  return x;
}

dmat softmax(const dmat& mat)
{
  dmat res(mat);
  double total = 0.0;

  for (size_t i = 0; i < res.rows; i++) {
    total += std::exp(res.data[i]);
  }

  for (size_t i = 0; i < res.rows; i++) {
    res.data[i] = (std::exp(res.data[i]) / total);
  }

  return res;
}

double crossEntropyLoss(const matrix<double>& pred, const matrix<double>& label)
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