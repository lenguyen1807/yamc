#include "activation.h"
#include "linear.h"
#include "matrix.h"

int main()
{
  nn::Linear ln(10, 2);
  nn::ReLU relu {};

  nn::matrix<float> x(10, 1);
  x.fill(4.0f);

  auto ln_f = ln.forward(x);
  auto relu_f = relu.forward(ln_f);

  ln_f.print();
  relu_f.print();

  return 0;
}