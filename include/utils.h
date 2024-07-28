#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <vector>
#include <cmath>

#ifdef _WIN32
#include "windows.h"
#include "psapi.h"
#endif

constexpr size_t EPOCHS = 5;

using MatrixPtr = std::shared_ptr<class Matrix>;
using ImagePtr = std::unique_ptr<class Image>;
using LayerPtr = std::unique_ptr<class Layer>;
using MatrixVector = std::vector<MatrixPtr>;
using ImageVector = std::vector<ImagePtr>;
using LayerVector = std::vector<LayerPtr>;

#ifdef _WIN32
// function to see how much memory used
void GetMemoryInfo();
#endif

double Sigmoid(double x);
double SigmoidGrad(double x);

double ReLU(double x);
double ReLUGrad(double x);

double Linear(double x);
double LinearGrad(double x);

MatrixPtr SoftMax(const MatrixPtr& x);

double CrossEntropyLoss(const class Matrix& pred, const class Matrix& label);

#endif // UTILS_H