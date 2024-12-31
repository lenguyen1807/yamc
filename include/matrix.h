#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
#  include <vecLib/cblas.h>
#  include <vecLib/cblas_new.h>
#else
// For Windows and Linux
#  include "cblas.h"
#endif

namespace nn
{

template<typename T>
class matrix
{
public:
  size_t rows;
  size_t cols;
  std::vector<T> data;

  matrix()
      : rows(0)
      , cols(0)
      , data()
  {
  }

  explicit matrix(size_t rows, size_t cols)
      : rows(rows)
      , cols(cols)
      , data(rows * cols)
  {
  }

  matrix(const matrix<T>& mat)
      : rows(mat.rows)
      , cols(mat.cols)
      , data(rows * cols)
  {
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = mat.data[i];
    }
  }

  // HACK: Cast constructor to use with static_cast
  template<typename NewType>
  matrix(const matrix<NewType>& mat)
      : rows(mat.rows)
      , cols(mat.cols)
      , data(rows * cols)
  {
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = static_cast<T>(mat.data[i]);
    }
  }

  // Why do we use this swap function ?
  // It is from copy-and-swap idiom
  friend void mat_swap(matrix<T>& first, matrix<T>& second)
  {
    using std::swap;
    swap(first.cols, second.cols);
    swap(first.rows, second.rows);
    swap(first.data, second.data);
  }

  matrix<T>& operator=(matrix<T> mat)
  {
    mat_swap(*this, mat);
    return *this;
  }

  matrix<T>& operator+=(const matrix<T>& mat)
  {
    check_dim(mat);
#ifdef __APPLE__
    catlas_saxpby(
        rows * cols, 1.0f, mat.data.data(), 1, 1.0f, this->data.data(), 1);
#else
#  pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] += mat.data[i];
    }
#endif
    return *this;
  }

  /* Unary minus operator */
  matrix<T> operator-() const
  {
    matrix<T> res(*this);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = -data[i];
    }
    return res;
  }

  /* This is element-wise multiplication */
  matrix<T>& operator%=(const matrix<T>& mat)
  {
    check_dim(mat);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] *= mat.data[i];
    }
    return *this;
  }

  matrix<bool> operator>(const T& value) const
  {
    matrix<bool> res(this->rows, this->cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = data[i] > value ? true : false;
    }
    return res;
  }

  matrix<bool> operator<(const T& value) const
  {
    matrix<bool> res(this->rows, this->cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = data[i] > value ? true : false;
    }
    return res;
  }

  /* Apply a function element wise
    - why use that ugly template, the reason is I want to pass a function with
    arbitrary number of parameters
    - ... is called variadic template
  */
  template<typename Func, typename... Args>
  matrix<T> apply(Func func, Args... args) const
  {
    matrix<T> new_mat(rows, cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      new_mat.data[i] = func(data[i], args...);
    }
    return new_mat;
  }

  /* Some ultility functions */
  void fill(const T& value)
  {
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = value;
    }
  }

  // Transpose function
  matrix<T> t() const
  {
    matrix<T> res(cols, rows);
#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        res.data[j * rows + i] = data[i * cols + j];
      }
    }
    return res;
  }

  void print()
  {
    std::cout << "Rows: " << rows << "\n";
    std::cout << "Cols: " << cols << "\n";
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        std::cout << data[i * cols + j] << " ";
      }
      std::cout << "\n";
    }
  }

  size_t arg_max() const
  {
    // find max element in array
    // https://stackoverflow.com/questions/73550037/finding-max-value-in-a-array
    T max_val = matrix<T>::max(*this);
    for (size_t i = 0; i < rows * cols; i++) {
      if (data[i] == max_val) {
        return i;
      }
    }
  }

  T reduce_sum()
  {
    T total {};
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      total += data[i];
    }
    return total;
  }

  matrix<T> sum(size_t axis)
  {
    assert(axis == 0 || axis == 1);

    // NOTE: Thanks claude for this, I should fix this in the future
    if (axis == 0) {
      matrix<T> res(1, cols);
#pragma omp parallel for
      for (size_t j = 0; j < cols; j++) {
        T total {};
        for (size_t i = 0; i < rows; i++) {
          total += data[i * cols + j];
        }
        res[j] = total;
      }
      return res;
    }

    if (axis == 1) {
      matrix<T> res(rows, 1);
#pragma omp parallel for
      for (size_t i = 0; i < rows; i++) {
        T total {};
        for (size_t j = 0; j < cols; j++) {
          total += data[i * cols + j];
        }
        res[i] = total;
      }
      return res;
    }
  }

  /* Static function */
  static matrix<T> flatten(const matrix<T>& mat)
  {
    matrix<T> res(mat);
    res.rows = mat.rows * mat.cols;
    res.cols = 1;
    return res;
  }

  static matrix<T> nrand(size_t rows, size_t cols, T mu, T std)
  {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::normal_distribution<T> distr(mu, std);

    matrix<T> res(rows, cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = distr(generator);
    }
    return res;
  }

  static matrix<T> urand(size_t rows, size_t cols, T range_from, T range_to)
  {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);

    matrix<T> res(rows, cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = distr(generator);
    }
    return res;
  }

  // Create this function for dropout
  static matrix<T> brand(size_t rows, size_t cols, float p)
  {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    // perform only 1 trial (will have two states 0 and 1)
    std::binomial_distribution<int> distr(1, p);

    matrix<T> res(rows, cols);
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = static_cast<T>(distr(generator));
    }
    return res;
  }

  static matrix<T> onehot(T value, std::vector<T> classes)
  {
    matrix<T> res(classes.size(), 1);
#pragma omp parallel for
    for (size_t i = 0; i < classes.size(); i++) {
      res.data[i] = classes[i] == value ? 1 : 0;
    }
    return res;
  }

  static matrix<T> values_like(const T& value, const matrix<T>& mat)
  {
    matrix<T> res(mat.rows, mat.cols);
    res.fill(value);
    return res;
  }

  static matrix<T> where(const matrix<bool> mask,
                         const matrix<T>& mat1,
                         const matrix<T> mat2)
  {
    mat1.check_dim(mat2);
    matrix<T> res(mat1);
#pragma omp parallel for
    for (size_t i = 0; i < res.rows * res.cols; i++) {
      res.data[i] = mask.data[i] ? mat1.data[i] : mat2.data[i];
    }
    return res;
  }

  static T max(const matrix<T>& mat)
  {
    // find max element in array
    // https://stackoverflow.com/questions/73550037/finding-max-value-in-a-array
    T max_val = *std::max_element(mat.data.begin(), mat.data.end());
    return max_val;
  }

private:
  void check_dim(const matrix<T>& mat) const
  {
    assert(mat.cols == cols);
    assert(mat.rows == rows);
  }
};

template<typename T>
inline matrix<T> operator+(matrix<T> left, const matrix<T>& right)
{
  left += right;
  return left;
}

template<typename T>
inline matrix<T> operator+(const matrix<T>& mat, const T& value)
{
  matrix<T> res(mat);
#pragma omp parallel for
  for (size_t i = 0; i < mat.rows * mat.cols; i++) {
    res.data[i] = mat.data[i] + value;
  }
  return res;
}

template<typename T>
inline matrix<T> operator-(const matrix<T>& mat, const T& value)
{
  return mat + (-value);
}

template<typename T>
inline matrix<T> operator-(const matrix<T>& left, const matrix<T>& right)
{
  return left + (-right);
}

template<typename T>
inline matrix<T> operator%(matrix<T> left, const matrix<T>& right)
{
  left %= right;
  return left;
}

/* Scalar multiplication */
template<typename T>
inline matrix<T> operator%(const matrix<T>& mat, const T& value)
{
  matrix<T> res(mat);
#ifdef __APPLE__
  cblas_sscal(res.rows * res.cols, value, res.data.data(), 1);
#else
#  pragma omp parallel for
  for (size_t i = 0; i < res.rows * res.cols; i++) {
    res.data[i] *= value;
  }
#endif
  return res;
}

template<typename T>
inline matrix<T> operator%(const T& value, const matrix<T>& mat)
{
  return mat % value;
}

/*
  Matrix multiplication. This is where we need optimization
  Reference:
  - https://siboehm.com/articles/22/Fast-MMM-on-CPU
*/
template<typename T>
inline matrix<T> operator*(const matrix<T>& left, const matrix<T>& right)
{
  assert(left.cols == right.rows);

  size_t rows = left.rows;
  size_t cols = right.cols;
  size_t inners = right.rows;  // can be left.cols

  matrix<T> res(rows, cols);

#ifdef __APPLE__
  /* Use Apple Accelerate framework to optimize */
  __LAPACK_int blas_rows = (__LAPACK_int)rows;
  __LAPACK_int blas_cols = (__LAPACK_int)cols;
  __LAPACK_int blas_inners = (__LAPACK_int)inners;

  // NOTE: I have to handle for multiple data type
  // sgemm only works for float data type
  cblas_sgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              blas_rows,
              blas_cols,
              blas_inners,
              1.0,
              left.data.data(),
              blas_inners,
              right.data.data(),
              blas_cols,
              0.0,
              res.data.data(),
              blas_cols);
#else
  /* Fall back to naive solution
  https://stackoverflow.com/questions/60360361/matrix-multiplication-using-openmp-c-collapsing-all-the-loops
  */
#  pragma omp parallel num_threads(omp_get_thread_num())
  {
#  pragma omp for
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        T prod {};
        for (size_t k = 0; k < inners; k++) {
          prod += left.data[i * inners + k] * right.data[k * cols + j];
        }
        res.data[i * cols + j] = prod;
      }
    }
  }
#endif

  return res;
}

}  // namespace nn

#endif  // MATRIX_H