#include "layers/activation.h"
#include "layers/linear.h"
#include "models/mlp.h"

MLP::MLP(size_t input_size, size_t output_size)
{
  /* Now add layer, using index for forward */
  add<0, nn::Linear>(input_size, 500);
  add<1, nn::ReLU>();
  add<2, nn::Linear>(500, 300);
  add<3, nn::ReLU>();
  add<4, nn::Linear>(300, 100);
  add<5, nn::ReLU>();
  add<6, nn::Linear>(100, output_size);
}

nn::matrix<float> MLP::forward(const nn::matrix<float>& input)
{
  nn::matrix<float> out(input);
  for (const auto& layer : layers) {
    out = layer.second->forward(out);
  }
  return out;
}

void MLP::backward(const nn::matrix<float>& grad)
{
  // just backward each layer
  nn::matrix<float> result(grad);
  for (auto iter = layers.rbegin(); iter != layers.rend(); ++iter) {
    result = iter->second->backward(result);
  }
}

void MLP::print_stats() {}
