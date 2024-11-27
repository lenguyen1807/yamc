#include <cmath>

#include "matrix.h"
#include "utils.h"

using namespace nn::F;

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