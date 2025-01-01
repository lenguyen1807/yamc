#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <vector>

#include "layer.h"
#include "optimizer.h"

namespace nn
{
// forward declaration
class Loss;
class Optimizer;

class Module
{
public:
  Module() = default;

  template<typename LayerType, typename... Args>
  void add(Args... args)
  {
    m_layers.emplace_back(new LayerType(args...));
  }

  matrix<float> forward(const matrix<float>& input);
  void backward(const matrix<float>& loss_grad);
  void zero_grad();
  void train();
  void eval();

  friend class Optimizer;
  friend class Loss;

private:
  std::vector<std::unique_ptr<Layer<float>>> m_layers = {};
  bool m_train = false;
};
}  // namespace nn

#endif  // MODEL_H