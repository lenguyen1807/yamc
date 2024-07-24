#include "utils.h"
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

MatrixType SoftMax(const MatrixType& x)
{
    MatrixType result(x.size(), std::vector<double>(1));
    double total = 0.0;

    for (size_t i = 0; i < x.size(); i++)
        total += std::exp(x[i][0]);        

    for (size_t i = 0; i < x.size(); i++)
    {
        std::vector<double> temp = {std::exp(x[i][0]) / total};
        result.emplace_back(temp);
    }

    return result;
}

double Accuracy(const MatrixType& pred, const MatrixType& label)
{
    // assume pred and label are vectors
    assert(pred.size() == label.size());
    assert(pred[0].size() == 1);
    assert(label[0].size() == 1);

    // calculate accuracy
    int eq = 0;
    for (size_t i = 0; i < pred.size(); i++)
    {
        if (pred[i][0] == label[i][0])
            eq += 1;
    }

    return eq / static_cast<double>(pred.size()); // or label.size()
}

double CrossEntropyLoss(const MatrixType& pred, const MatrixType& label, size_t numClass)
{
    assert(pred.size() == label.size());
    assert(pred.size() == numClass);

    // first apply softmax to predict
    MatrixType predSMax = SoftMax(pred);

    // calculate cross entropy loss
    double entropy = 0.0;
    for (size_t i = 0; i < numClass; i++)
    {
        entropy += label[i][0] * std::log(pred[i][0]);
    }
    return (-entropy);
}