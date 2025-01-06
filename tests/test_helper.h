#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <cmath>

#include "matrix.h"

#define EPSILON_STD(T) std::numeric_limits<T>::epsilon()
#define EPSILON_FLT 0.001f

// https://stackoverflow.com/questions/4548004/how-to-correctly-and-standardly-compare-floats
template<typename T>
static bool are_equal(T f1, T f2, T epsilon)
{
  return (std::fabs(f1 - f2)
          <= epsilon * std::fmax(std::fabs(f1), std::fabs(f2)));
}

inline bool is_close_mat(const nn::matrix<float>& mat1,
                         const nn::matrix<float>& mat2,
                         float epsilon)
{
  if ((mat1.rows != mat2.rows) || (mat1.cols != mat2.cols)) {
    return false;
  }

  for (size_t i = 0; i < mat1.rows * mat2.cols; i++) {
    if (!are_equal<float>(mat1.data[i], mat2.data[i], epsilon)) {
      return false;
    }
  }

  return true;
}

#endif  // TEST_HELPER_H