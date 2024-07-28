#include "utils.h"
#include "matrix.h"
#include <cassert>
#include <iostream>

double Sigmoid(double x)
{
    return (1.0 / (1 + std::exp(-x)));
}

double SigmoidGrad(double x)
{
    return Sigmoid(x) * (1.0 - Sigmoid(x));
}

double ReLU(double x)
{
    return x > 0.0 ? x : 0.0;
}

double ReLUGrad(double x)
{
    return x > 0.0 ? 1 : 0.0;
}

double Linear(double x)
{
    return x;
}

double LinearGrad(double x)
{
    return 1.0;
}

MatrixPtr SoftMax(const MatrixPtr& x)
{
    // suppose x is a column vector
    assert(x->cols == 1);

    Matrix result(x->rows, 1);
    double total = 0.0;

    for (size_t i = 0; i < x->rows; i++)
        total += std::exp(x->values[i][0]);        

    for (size_t i = 0; i < x->rows; i++)
    {
        result.values[i][0] = (std::exp(x->values[i][0]) / total);
    }

    return std::make_shared<Matrix>(result);
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

#ifdef _WIN32
void GetMemoryInfo()
{
    // ref: https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

    std::cout << "Used Memory: " << physMemUsedByMe / 1000000.0 << " MB \n"
              << "-------------------------------------------\n";
}
#endif