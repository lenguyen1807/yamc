#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <functional>
#include "utils.h"

class Matrix
{
public:
    size_t rows;
    size_t cols;
    std::vector<std::vector<double>> values;

    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix& mat);

    void Print();

    // Fill all element in matrix with one value
    void Fill(double value);

    static void CheckDimension(const Matrix& mat1, const Matrix& mat2);

    // return a column vector for matrix, e.g. (3, 4) -> (7, 1)
    static Matrix Flatten(const Matrix& mat);

    // random matrix 
    static Matrix Randomized(size_t rows, size_t cols);

    // create one-hot vector
    static Matrix OneHot(size_t value, size_t classes);

    // get index of max value in a vector
    double Max() const;
    size_t ArgMax() const;

    Matrix& operator=(const Matrix& mat);

    // Add matrix
    Matrix& operator+=(const Matrix& mat);
    // Subtract matrix
    Matrix& operator-=(const Matrix& mat);
    // Element wise multiply
    Matrix& operator%=(const Matrix& mat);
    // Transpose matrix
    Matrix T();

    // Apply a function to matrix
    Matrix Apply(const std::function<double(double)>& func);
};

// Add matrix
inline Matrix operator+(Matrix mat1, const Matrix& mat2)
{
    mat1 += mat2;
    return mat1;
}

// Subtract matrix
inline Matrix operator-(Matrix mat1, const Matrix& mat2)
{
    mat1 -= mat2;
    return mat1;
}

// Element wise multiply matrix
inline Matrix operator%(Matrix mat1, const Matrix& mat2)
{
    mat1 %= mat2;
    return mat1;
}

// Multiply two matrix
inline Matrix operator*(const Matrix& mat1, const Matrix& mat2)
{
    assert(mat1.cols == mat2.rows);
    
    Matrix res(mat1.rows, mat2.cols);

    for (size_t i = 0; i < mat1.rows; i++)
    {
        for (size_t j = 0; j < mat2.cols; j++)
        {
            for (size_t k = 0; k < mat1.cols; k++)
            {
                res.values[i][j] += mat1.values[i][k]*mat2.values[k][j];
            }
        }
    }

    return res;
}

// Scalar multiplication
inline Matrix operator*(double value, const Matrix& mat)
{
    Matrix res(mat.rows, mat.cols);

    for (size_t i = 0; i < mat.rows; i++)
    {
        for (size_t j = 0; j < mat.cols; j++)
        {
            res.values[i][j] = value * mat.values[i][j];
        }
    }
    return res;
}

#endif // MATRIX_H