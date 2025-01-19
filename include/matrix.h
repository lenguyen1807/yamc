#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
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

  explicit matrix(const std::vector<std::vector<T>>& input)
      : rows(input.size())
      , cols(rows != 0 ? input[0].size() : 0)
      , data(rows * cols)
  {
    for (size_t i = 0; i < input.size(); i++) {
      for (size_t j = 0; j < input[i].size(); j++) {
        data[i * cols + j] = input[i][j];
      }
    }
  }

  matrix(const matrix<T>& mat)
      : rows(mat.rows)
      , cols(mat.cols)
      , data(rows * cols)
  {
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
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] += mat.data[i];
    }
    return *this;
  }

  /* Unary minus operator */
  matrix<T> operator-() const
  {
    matrix<T> res(*this);
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = -data[i];
    }
    return res;
  }

  /* This is element-wise multiplication */
  matrix<T>& operator%=(const matrix<T>& mat)
  {
    check_dim(mat);
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] *= mat.data[i];
    }
    return *this;
  }

  matrix<bool> operator>(const T& value) const
  {
    matrix<bool> res(this->rows, this->cols);
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = data[i] > value ? true : false;
    }
    return res;
  }

  matrix<bool> operator<(const T& value) const
  {
    matrix<bool> res(this->rows, this->cols);
    for (size_t i = 0; i < rows * cols; i++) {
      res.data[i] = data[i] < value ? true : false;
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
    for (size_t i = 0; i < rows * cols; i++) {
      new_mat.data[i] = func(data[i], args...);
    }
    return new_mat;
  }

  /* Some ultility functions */
  void fill(const T& value) { std::fill(data.begin(), data.end(), value); }

  // Transpose function
  matrix<T> t() const
  {
    matrix<T> res(cols, rows);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        res.data[j * rows + i] = data[i * cols + j];
      }
    }
    return res;
  }

  void print() const
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
    return -1;
  }

  T reduce_sum() const
  {
    T total {};
#pragma omp parallel for reduction(+ : total)
    for (size_t i = 0; i < rows * cols; i++) {
      total += data[i];
    }
    return total;
  }

  matrix<T> sum(size_t axis)
  {
    // sum by row
    if (axis == 0) {
      matrix<T> res(1, cols);
      for (size_t j = 0; j < cols; j++) {
        T total {};
#pragma omp parallel for reduction(+ : total)
        for (size_t i = 0; i < rows; i++) {
          total += data[i * cols + j];
        }
        res.data[j] = total;
      }
      return res;
    } else if (axis == 1) {
      matrix<T> res(rows, 1);
      for (size_t i = 0; i < rows; i++) {
        T total {};
#pragma omp parallel for reduction(+ : total)
        for (size_t j = 0; j < cols; j++) {
          total += data[i * cols + j];
        }
        res.data[i] = total;
      }
      return res;
    }
    throw std::invalid_argument("Only accept axis=0 or axis=1");
  }

  matrix<T> broadcast_col(size_t new_cols)
  {
    if (cols > new_cols) {
      throw std::invalid_argument(
          "New columns should be larger than old columns");
    }

    matrix<float> res(*this);
    while (res.cols < new_cols) {
      res = hstack(res, *this);
    }
    return res;
  }

  matrix<T> broadcast_row(size_t new_rows)
  {
    if (rows > new_rows) {
      throw std::invalid_argument("New rows should be larger than old rows");
    }

    matrix<float> res(*this);
    while (res.rows < new_rows) {
      res = vstack(res, *this);
    }
    return res;
  }

  matrix<T> mean(size_t axis = 0)
  {
    if (axis == 0) {
      return this->sum(0) / static_cast<T>(rows);
    } else if (axis == 1) {
      return this->sum(1) / static_cast<T>(cols);
    } else {
      throw std::invalid_argument("Only support axis = 1 or axis = 0");
    }
  }

  matrix<T> max_by_row()
  {
    matrix<T> res(rows, 1);
    for (size_t i = 0; i < rows; i++) {
      // max element is always first element of rows
      T max_elm = std::numeric_limits<T>::min();
      for (size_t j = 0; j < cols; j++) {
        if (data[i * cols + j] > max_elm) {
          max_elm = data[i * cols + j];
        }
      }
      res.data[i] = max_elm;
    }
    return res;
  }

  matrix<T> max(size_t axis = 0)
  {
    if (axis == 0) {
      // HACK: This will decrease performance so far
      // Need to reimplement but I'm too lazy
      return this->t().max_by_row().t();
    } else if (axis == 1) {
      return this->max_by_row();
    } else {
      throw std::invalid_argument("Only support axis = 1 or axis = 0");
    }
  }

  bool empty() const { return cols == 0 && rows == 0 && data.empty(); }

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
    matrix<T> res(rows, cols);
#pragma omp parallel
    {
      // Each thread gets its own generator
      std::random_device rand_dev;
      std::mt19937 generator(rand_dev() + omp_get_thread_num());
      std::normal_distribution<T> distr(mu, std);

#pragma omp for
      for (size_t i = 0; i < rows * cols; i++) {
        res.data[i] = distr(generator);
      }
    }
    return res;
  }

  static matrix<T> urand(size_t rows, size_t cols, T range_from, T range_to)
  {
    matrix<T> res(rows, cols);
#pragma omp parallel
    {
      // Each thread gets its own generator
      std::random_device rand_dev;
      std::mt19937 generator(rand_dev() + omp_get_thread_num());
      std::uniform_real_distribution<T> distr(range_from, range_to);

#pragma omp for
      for (size_t i = 0; i < rows * cols; i++) {
        res.data[i] = distr(generator);
      }
    }
    return res;
  }

  // Create this function for dropout
  static matrix<T> brand(size_t rows, size_t cols, float p)
  {
    matrix<T> res(rows, cols);
#pragma omp parallel
    {
      // Each thread gets its own generator
      std::random_device rand_dev;
      std::mt19937 generator(rand_dev() + omp_get_thread_num());
      std::binomial_distribution<int> distr(1, p);

#pragma omp for
      for (size_t i = 0; i < rows * cols; i++) {
        res.data[i] = distr(generator);
      }
    }
    return res;
  }

  static matrix<T> onehot(T value, std::vector<T> classes)
  {
    matrix<T> res(classes.size(), 1);
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

  static T min(const matrix<T>& mat)
  {
    T min_val = *std::min_element(mat.data.begin(), mat.data.end());
    return min_val;
  }

  static T norm(const matrix<T>& mat)
  {
    // we want a vector
    if (mat.cols != 1) {
      throw std::invalid_argument("Only a vector can have norm value");
    }

    T norm {};
#pragma omp parallel for reduction(+ : norm)
    for (size_t i = 0; i < mat.rows * mat.cols; i++) {
      norm += (mat.data[i] * mat.data[i]);
    }
    return norm;
  }

  static matrix<T> vstack(const matrix<T>& mat1, const matrix<T>& mat2)
  {
    // just a trick
    if (mat1.empty()) {
      return mat2;
    }

    if (mat1.cols != mat2.cols) {
      throw std::invalid_argument("Must be equal in columns");
    }

    size_t new_rows = mat1.rows + mat2.rows;
    matrix<T> res(new_rows, mat1.cols);

    size_t i, j;

    // We will stack two time
#pragma omp parallel private(i, j)
    for (i = 0; i < mat1.rows; i++) {
      for (j = 0; j < mat1.cols; j++) {
        res.data[i * mat1.cols + j] = mat1.data[i * mat1.cols + j];
      }
    }

#pragma omp parallel private(i, j)
    for (i = 0; i < mat2.rows; i++) {
      for (j = 0; j < mat2.cols; j++) {
        res.data[(i + mat1.rows) * mat2.cols + j] =
            mat2.data[i * mat2.cols + j];
      }
    }

    return res;
  }

  static matrix<T> hstack(const matrix<T>& mat1, const matrix<T>& mat2)
  {
    // just a trick
    if (mat1.empty()) {
      return mat2;
    }

    if (mat1.rows != mat2.rows) {
      throw std::invalid_argument("Must be equal in rows");
    }

    size_t new_cols = mat1.cols + mat2.cols;
    matrix<T> res(mat1.rows, new_cols);

    size_t i, j;

    // We will stack two time
#pragma omp parallel private(i, j)
    for (i = 0; i < mat1.rows; i++) {
      for (j = 0; j < mat1.cols; j++) {
        res.data[i * (new_cols) + j] = mat1.data[i * mat1.cols + j];
      }
    }

#pragma omp parallel private(i, j)
    for (i = 0; i < mat2.rows; i++) {
      for (j = 0; j < mat2.cols; j++) {
        res.data[i * (new_cols) + (j + mat1.cols)] =
            mat2.data[i * mat2.cols + j];
      }
    }

    return res;
  }

  // return a column vector with value from start -> end-1
  static matrix<T> arange(const T& start, const T& end)
  {
    size_t range = std::abs(start - end);
    matrix<T> res(range, 1);

    for (size_t i = 0; i < range; i++) {
      res.data[i] = start + i;
    }

    return res;
  };

private:
  void check_dim(const matrix<T>& mat) const
  {
    if (mat.cols != cols || mat.rows != rows) {
      throw std::invalid_argument("Two matrix should be equal in size");
    }
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
  for (size_t i = 0; i < res.rows * res.cols; i++) {
    res.data[i] *= value;
  }
  return res;
}

template<typename T>
inline matrix<T> operator%(const T& value, const matrix<T>& mat)
{
  return mat % value;
}

template<typename T>
inline matrix<T> operator/(const matrix<T>& mat, const T& value)
{
  matrix<T> res(mat);
  for (size_t i = 0; i < res.rows * res.cols; i++) {
    res.data[i] /= value;
  }
  return res;
}

/*
  Matrix multiplication. This is where we need optimization
  Reference:
  - https://siboehm.com/articles/22/Fast-MMM-on-CPU
*/
template<typename T>
inline matrix<T> operator*(const matrix<T>& left, const matrix<T>& right)
{
  if (left.cols != right.rows) {
    throw std::invalid_argument("Left columns should equal right rows");
  }

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
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      T prod {};
      for (size_t k = 0; k < inners; k++) {
        prod += left.data[i * inners + k] * right.data[k * cols + j];
      }
      res.data[i * cols + j] = prod;
    }
  }
#endif

  return res;
}

}  // namespace nn

#endif  // MATRIX_H