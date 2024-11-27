#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "cblas.h"

template<typename T>
class matrix
{
public:
  size_t rows;
  size_t cols;
  T* data;

  matrix() = delete;
  matrix(size_t rows, size_t cols);
  matrix(const matrix<T>& mat);
  ~matrix();

  // Some methods to create special matrix
  static auto flatten(const matrix<T>& mat) -> matrix<T>;
  // random with normal distribution
  static auto nrand(size_t rows, size_t cols, T mu, T std) -> matrix<T>;
  // random with uniform distribution
  static auto urand(size_t rows, size_t cols, T range_from, T range_to)
      -> matrix<T>;
  static auto onehot(size_t value, std::vector<T> classes) -> matrix<T>;

  // Some operators
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
  auto t() -> matrix<T>;

  // fill value
  void fill(T value);

  // apply a function element wise
  // why use that ugly template, the reason is I want to pass a function
  // with arbitrary number of parameters
  // ... is called variadic template
  template<typename Func, typename... Args>
  auto apply(Func&& func, Args&&... args) -> matrix<T>
  {
    matrix<T> new_mat(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
      new_mat.data[i] = func(data[i], std::forward<Args>(args)...);
    }
    return new_mat;
  }

  // print function
  void print();

  void check_dim(const matrix<T>& mat)
  {
    assert(rows == mat.rows && cols == mat.cols);
  }
};

template<typename T>
matrix<T>::~matrix()
{
  delete[] this->data;
}

template<typename T>
matrix<T>::matrix(size_t rows, size_t cols)
    : rows {rows}
    , cols {cols}
    , data {new T[rows * cols] {}}
{
}

template<typename T>
matrix<T>::matrix(const matrix<T>& mat)
    : rows {mat.rows}
    , cols {mat.cols}
    , data {new T[rows * cols] {}}
{
  for (size_t i = 0; i < rows * cols; i++) {
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
  for (size_t i = 0; i < rows * cols; i++) {
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
  for (size_t i = 0; i < rows * cols; i++) {
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
  for (size_t i = 0; i < rows * cols; i++) {
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
inline auto operator*(const matrix<T>& mat, const T& value) -> matrix<T>
{
  matrix<T> res(mat);
  for (size_t i = 0; i < res.rows * res.cols; i++) {
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

  size_t cols = right.cols;
  size_t rows = left.rows;
  size_t inners = right.rows;  // can be left.cols
  matrix<T> res(rows, cols);

#ifndef GEMM_OPT
  // naive implementation
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      T prod {};
      for (size_t k = 0; k < inners; k++) {
        prod += left.data[i * cols + k] * right.data[k * cols + j];
      }
      res.data[i * cols + j] = prod;
    }
  }
#else
  if (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                left.rows,
                right.cols,
                left.cols,
                1.0f,
                left.data,
                left.cols,
                right.data,
                right.cols,
                0.0f,
                res.data,
                res.cols);
  } else if (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                left.rows,
                right.cols,
                left.cols,
                1.0,
                left.data,
                left.cols,
                right.data,
                right.cols,
                0.0,
                res.data,
                res.cols);
  }
#endif
  return res;
}

// --- Function implementation ---

template<typename T>
void matrix<T>::fill(T value)
{
  for (size_t i = 0; i < rows * cols; i++) {
    data[i] = value;
  }
}

template<typename T>
auto matrix<T>::t() -> matrix<T>
{
  matrix<T> res(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      res.data[j * rows + i] = data[cols * i + j];
    }
  }
  return res;
}

template<typename T>
void matrix<T>::print()
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
auto matrix<T>::onehot(size_t value, std::vector<T> classes) -> matrix<T>
{
  matrix<T> res(classes.size(), 1);
  for (size_t i = 0; i < classes.size(); i++) {
    res.data[i] = classes[i] == value ? 1 : 0;
  }
  return res;
}

template<typename T>
auto matrix<T>::nrand(size_t rows, size_t cols, T mu, T std) -> matrix<T>
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::normal_distribution<T> distr(mu, std);

  matrix<T> res(rows, cols);
  for (size_t i = 0; i < rows * cols; i++) {
    res.data[i] = distr(generator);
  }
  return res;
}

template<typename T>
auto matrix<T>::urand(size_t rows, size_t cols, T range_from, T range_to)
    -> matrix<T>
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<T> distr(range_from, range_to);

  matrix<T> res(rows, cols);
  for (size_t i = 0; i < rows * cols; i++) {
    res.data[i] = distr(generator);
  }
  return res;
}

// some convenient aliases

using dmat = matrix<double>;
using imat = matrix<int>;
using fmat = matrix<float>;
using lmat = matrix<long>;

#endif  // MATRIX_H