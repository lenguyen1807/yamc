#include <cmath>
#include <memory>

#include "layer.h"
#include "matrix.h"
#include "utils.h"

nn::Linear::Linear(size_t input_size,
                   size_t output_size,
                   activation activ,
                   bool randInit)
    : pre_activ(nullptr)
    , post_activ(nullptr)
    , weight(nullptr)
    , grad(nullptr)
    , weight_grad(nullptr)
{
  if (randInit) {
    weight = std::make_unique<dmat>(dmat::nrand(
        output_size, input_size, 0.0, 2.0 / std::sqrt(output_size)));
  } else {
    weight = std::make_unique<dmat>(output_size, input_size);
    weight->fill(0.0);
  }

  switch (activ) {
    case nn::activation::LINEAR:
      activ_func = nn::F::linear;
      break;
    case nn::activation::RELU:
      activ_func = nn::F::relu;
      break;
    case nn::activation::SIGMOID:
      activ_func = nn::F::sigmoid;
      break;
  }
}

void nn::Linear::forward(const std::unique_ptr<dmat>& input)
{
  if (pre_activ == nullptr) {
    pre_activ = std::make_unique<dmat>((*weight) * (*input));
  } else {
    (*pre_activ) = (*weight) * (*input);
  }

  if (post_activ != nullptr) {
    post_activ = std::make_unique<dmat>(pre_activ->apply(activ_func, false));
  } else {
    (*post_activ) = pre_activ->apply(activ_func, false);
  }
}

void nn::Linear::backward(const std::unique_ptr<matrix<double>>& prevGrad,
                          const std::unique_ptr<matrix<double>>& prevLayer)
{
  // step 1: compute layer after activation
  dmat res = (*prevGrad) % (pre_activ->apply(activ_func, true));

  // step 2: compute gradient respect to weight
  *weight_grad = res * (prevLayer->t());

  // step 3: compute pre-activation layer
  *grad = weight->t() * res;
}

void nn::Linear::zero_grad()
{
  grad->fill(0.0);
}