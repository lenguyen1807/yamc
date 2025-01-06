#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <memory>

#include "layers/layer.h"
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
  virtual ~Module() = default;

  template<size_t index, typename LayerType, typename... Args>
  void add(Args... args)
  {
    layers[index] = std::make_unique<LayerType>(args...);
  }

  /*
  NOTE: Each model derived from module need to implement forward and backward
  itself
  */

  void zero_grad();
  void train();
  void eval();

  friend class Optimizer;
  friend class Loss;

protected:
  std::map<size_t, std::unique_ptr<Layer<float>>> layers = {};
};
}  // namespace nn

#endif  // MODEL_H