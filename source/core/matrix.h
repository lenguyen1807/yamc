#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include <omp.h>

#ifdef GEMM_OPT
#  include "cblas.h"
#endif

namespace nn
{

template<typename T>
class matrix
{
public:
  int rows;
  int cols;
  std::vector<T> data;

  matrix();
  matrix(int rows, int cols);
  matrix(const matrix<T>& mat);

  // Some methods to create special matrix
  static auto flatten(const matrix<T>& mat) -> matrix<T>;
  // random with normal distribution
  static auto nrand(int rows, int cols, T mu, T std) -> matrix<T>;
  // random with uniform distribution
  static auto urand(int rows, int cols, T range_from, T range_to) -> matrix<T>;
  static auto onehot(int value, std::vector<T> classes) -> matrix<T>;

  auto operator=(matrix<T> mat) -> matrix<T>&;

  // Why do we use this swap function ?
  // It is from copy-and-swap idiom
  friend void mat_swap(matrix<T>& first, matrix<T>& second)
  {
    using std::swap;
    swap(first.cols, second.cols);
    swap(first.rows, second.rows);
    swap(first.data, second.data);
  }

  auto operator+=(const matrix<T>& mat) -> matrix<T>&;
  auto operator-=(const matrix<T>& mat) -> matrix<T>&;
  auto operator%=(const matrix<T>& mat)
      -> matrix<T>&;  // this is element-wise multiply

  // matrix transpose
  auto t() const -> matrix<T>;

  // fill value
  void fill(T value);

  auto arg_max() const -> int;

  // apply a function element wise
  // why use that ugly template, the reason is I want to pass a function
  // with arbitrary number of parameters
  // ... is called variadic template
  template<typename Func, typename... Args>
  auto apply(Func func, Args... args) const -> matrix<T>
  {
    matrix<T> new_mat(rows, cols);
#pragma omp parallel for
    for (int i = 0; i < rows * cols; i++) {
      new_mat.data[i] = func(data[i], args...);
    }
    return new_mat;
  }

  // print function
  void print() const;

  void check_dim(const matrix<T>& mat)
  {
    assert(rows == mat.rows && cols == mat.cols);
  }
};

template<typename T>
matrix<T>::matrix()
    : rows {0}
    , cols {0}
    , data {}
{
}

template<typename T>
matrix<T>::matrix(int rows, int cols)
    : rows {rows}
    , cols {cols}
    , data(rows * cols)
{
}

template<typename T>
matrix<T>::matrix(const matrix<T>& mat)
    : rows {mat.rows}
    , cols {mat.cols}
    , data(mat.rows * mat.cols)
{
  for (int i = 0; i < rows * cols; i++) {
    data[i] = mat.data[i];
  }
}

// --- Operator implementation ---

template<typename T>
auto matrix<T>::operator=(matrix<T> mat) -> matrix<T>&
{
  mat_swap(*this, mat);
  return *this;
}

template<typename T>
auto matrix<T>::operator+=(const matrix<T>& mat) -> matrix<T>&
{
  this->check_dim(mat);
#pragma omp parallel for
  for (int i = 0; i < rows * cols; i++) {
    data[i] += mat.data[i];
  }
  return *this;
}

template<typename T>
inline auto operator+(matrix<T> left, const matrix<T>& right) -> matrix<T>
{
  left += right;
  return left;
}

template<typename T>
auto matrix<T>::operator-=(const matrix<T>& mat) -> matrix<T>&
{
  this->check_dim(mat);
#pragma omp parallel for
  for (int i = 0; i < rows * cols; i++) {
    data[i] -= mat.data[i];
  }
  return *this;
}

template<typename T>
inline auto operator-(matrix<T> left, const matrix<T>& right) -> matrix<T>
{
  left -= right;
  return left;
}

template<typename T>
auto matrix<T>::operator%=(const matrix<T>& mat) -> matrix<T>&
{
  this->check_dim(mat);
#pragma omp parallel for
  for (int i = 0; i < rows * cols; i++) {
    data[i] *= mat.data[i];
  }
  return *this;
}

template<typename T>
inline auto operator%(matrix<T> left, const matrix<T>& right) -> matrix<T>
{
  left %= right;
  return left;
}

// scalar multiplication
template<typename T>
inline auto operator%(const matrix<T>& mat, const T& value) -> matrix<T>
{
  matrix<T> res(mat);
#pragma omp parallel for
  for (int i = 0; i < res.rows * res.cols; i++) {
    res.data[i] *= value;
  }
  return res;
}

// matrix multiplication
// https://siboehm.com/articles/22/Fast-MMM-on-CPU
template<typename T>
inline auto operator*(const matrix<T>& left, const matrix<T>& right)
    -> matrix<T>
{
  assert(left.cols == right.rows);

  int rows = left.rows;
  int cols = right.cols;
  int inners = right.rows;  // can be left.cols

  matrix<T> res(rows, cols);

#ifndef GEMM_OPT
  // naive implementation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      T prod {};
      for (int k = 0; k < inners; k++) {
        prod += left.data[i * inners + k] * right.data[k * cols + j];
      }
      res.data[i * cols + j] = prod;
    }
  }
#else
  cblas_dgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              rows,
              cols,
              inners,
              1.0,
              left.data.data(),
              inners,
              right.data.data(),
              cols,
              0.0,
              res.data.data(),
              cols);
#endif
  return res;
}

// --- Function implementation ---

template<typename T>
void matrix<T>::fill(T value)
{
  for (int i = 0; i < rows * cols; i++) {
    data[i] = value;
  }
}

template<typename T>
auto matrix<T>::t() const -> matrix<T>
{
  matrix<T> res(cols, rows);
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      res.data[j * rows + i] = data[i * cols + j];
    }
  }
  return res;
}

template<typename T>
void matrix<T>::print() const
{
  std::cout << "Rows: " << rows << "\n";
  std::cout << "Cols: " << cols << "\n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << data[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}

// --- static function implementation ---

template<typename T>
auto matrix<T>::flatten(const matrix<T>& mat) -> matrix<T>
{
  matrix<T> res(mat);
  res.rows = mat.rows * mat.cols;
  res.cols = 1;
  return res;
}

template<typename T>
auto matrix<T>::onehot(int value, std::vector<T> classes) -> matrix<T>
{
  matrix<T> res(classes.size(), 1);
  for (int i = 0; i < classes.size(); i++) {
    res.data[i] = classes[i] == value ? 1 : 0;
  }
  return res;
}

template<typename T>
auto matrix<T>::nrand(int rows, int cols, T mu, T std) -> matrix<T>
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::normal_distribution<T> distr(mu, std);

  matrix<T> res(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    res.data[i] = distr(generator);
  }
  return res;
}

template<typename T>
auto matrix<T>::urand(int rows, int cols, T range_from, T range_to) -> matrix<T>
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<T> distr(range_from, range_to);

  matrix<T> res(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    res.data[i] = distr(generator);
  }
  return res;
}

template<typename T>
auto matrix<T>::arg_max() const -> int
{
  // find max element in array
  // https://stackoverflow.com/questions/73550037/finding-max-value-in-a-array
  T max_val = *std::max_element(data.begin(), data.end());

  for (int i = 0; i < rows * cols; i++) {
    if (data[i] == max_val) {
      return i;
    }
  }
}

// some convenient aliases
using dmat = matrix<double>;

}  // namespace nn

#endif  // MATRIX_H