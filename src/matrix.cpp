#include "matrix.h"
#include <iostream>
#include <random>

// Initialize a zero matrix
Matrix::Matrix(size_t rows, size_t cols)
: rows(rows)
, cols(cols)
, values(rows, std::vector<double>(cols))
{}

Matrix::Matrix(const Matrix& mat)
: rows(mat.rows)
, cols(mat.cols)
, values(rows, std::vector<double>(cols))
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            values[i][j] = mat.values[i][j];
        }
    }
}

void Matrix::Print()
{
    std::cout << "Rows: " << rows << "\n";
    std::cout << "Cols: " << cols << "\n";
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            std::cout << values[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::Fill(double value)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            values[i][j] = value;
        }
    }
}

void Matrix::CheckDimension(const Matrix& mat1, const Matrix& mat2)
{
    assert(mat1.rows == mat2.rows && mat2.cols == mat2.cols);
}

Matrix Matrix::Flatten(const Matrix& mat)
{
    Matrix res(mat.cols * mat.rows, 1);
    for (size_t i = 0; i < mat.rows; i++)
    {
        for (size_t j = 0; j < mat.cols; j++)
        {
            res.values[i*mat.rows + j][0] = mat.values[i][j];
        }
    }
    return res;
}

Matrix Matrix::Randomized(size_t rows, size_t cols)
{
    // generate random in uniform distribution
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());

    // https://cs231n.github.io/neural-networks-2/#init
    double std = 2.0f / static_cast<double>(cols); 
    std::normal_distribution<double> distr(0, std);

    Matrix res(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            res.values[i][j] = distr(generator);
        }
    }

    return res;
}

Matrix Matrix::OneHot(size_t value, size_t classes)
{
    Matrix res(classes, 1);
    for (size_t i = 0; i < classes; i++)
    {
        if (i == value)
            res.values[i][0] = 1;
    }
    return res;
}

double Matrix::Max() const
{
    assert(cols == 1);
    double max = values[0][0];
    for (size_t i = 1; i < rows; i++)
    {
        if (values[i][0] > max)
            max = values[i][0];
    }
    return max;
}

size_t Matrix::ArgMax() const
{
    double max = Max();
    for (size_t i = 0; i < rows; i++)
    {
        if (values[i][0] == max)
            return i;
    }
    return 0;
}

Matrix& Matrix::operator=(const Matrix& mat)
{
    this->cols = mat.cols;
    this->rows = mat.rows;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            values[i][j] = mat.values[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& mat)
{
    CheckDimension((*this), mat);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            this->values[i][j] -= mat.values[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator+=(const Matrix& mat)
{
    CheckDimension((*this), mat);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            this->values[i][j] += mat.values[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator%=(const Matrix& mat)
{
    CheckDimension((*this), mat);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            this->values[i][j] *= mat.values[i][j];
        }
    }
    return *this;
}

Matrix Matrix::T()
{
    Matrix mat(cols, rows);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            mat.values[j][i] = this->values[i][j];
        }
    }
    return mat;
}

Matrix Matrix::Apply(const std::function<double(double)>& func)
{
    Matrix mat(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            mat.values[i][j] = func(this->values[i][j]);
        }
    }
    return mat;
}