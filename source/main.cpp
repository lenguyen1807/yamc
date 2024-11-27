#define GEMM_OPT

#include "matrix.h"
#include "utils.h"

int main()
{
  dmat mat1 = dmat::nrand(3, 3, 0.0, 1.0);
  mat1.print();
  dmat mat2 = mat1.apply(nn::F::relu, false);
  mat2.print();
  dmat mat3 = mat1 * mat2;
  mat3.print();
  return 0;
}
