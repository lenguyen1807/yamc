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

class CrossEntropyLoss : public Loss<double>
{
public:
  CrossEntropyLoss(MLP* model);

  void backward(const matrix<double>& label) override;
  double operator()(const matrix<double>& logits,
                    const matrix<double>& label) override;

  const matrix<double>& get_pred() const;

private:
  MLP* m_pmodel;
  matrix<double> m_pred;
};
};  // namespace nn

#endif  // LOSS_H