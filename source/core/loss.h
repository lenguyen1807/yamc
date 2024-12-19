#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

namespace nn
{
class MLP;

template<typename T>
class Loss
{
public:
  virtual void backward(const matrix<T>& label) = 0;
  virtual T operator()(const matrix<T>& logits, const matrix<T>& label) = 0;
};

class CrossEntropyLoss : public Loss<float>
{
public:
  CrossEntropyLoss(MLP* model);

  void backward(const matrix<float>& label) override;
  float operator()(const matrix<float>& logits,
                   const matrix<float>& label) override;

  const matrix<float>& get_pred() const;

private:
  MLP* m_pmodel;
  matrix<float> m_pred;
};
};  // namespace nn

#endif  // LOSS_H