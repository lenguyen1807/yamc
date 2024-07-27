#include "utils.h"
#include "matrix.h"
#include <cassert>
#include <iostream>

double Sigmoid(double x) 
{
    return (1 / std::exp(-x));
}

double ReLU(double x) 
{
    return x > 0 ? x : 0;
}

double Linear(double x)
{
    return x;
}

Matrix SoftMax(const Matrix& x)
{
    // suppose x is a column vector
    assert(x.cols == 1);

    Matrix result(x.rows, 1);
    double total = 0.0;

    for (size_t i = 0; i < x.rows; i++)
        total += std::exp(x.values[i][0]);        

    for (size_t i = 0; i < x.rows; i++)
    {
        result.values[i][0] = (std::exp(x.values[i][0]) / total);
    }

    return result;
}

double CrossEntropyLoss(const Matrix& pred, const Matrix& label)
{
    assert(pred.cols == 1);
    assert(label.cols == 1);
    assert(pred.rows == label.rows);

    // calculate cross entropy loss
    double entropy = 0.0;
    for (size_t i = 0; i < label.rows; i++)
    {
        entropy += label.values[i][0] * std::log(pred.values[i][0]);
    }
    return (-entropy);
}