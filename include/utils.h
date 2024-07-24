#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <vector>
#include <cmath>

using MatrixPtr = std::shared_ptr<class Matrix>;
using ImagePtr = std::unique_ptr<class Image>;
using LayerPtr = std::unique_ptr<class Layer>;
using MatrixVector = std::vector<MatrixPtr>;
using ImageVector = std::vector<ImagePtr>;
using LayerVector = std::vector<LayerPtr>;
using MatrixType = std::vector<std::vector<double>>;

double Sigmoid(double x);
double ReLU(double x);
double Linear(double x);
MatrixType SoftMax(const MatrixType& x);
double Accuracy(const MatrixType& pred, const MatrixType& label);
double CrossEntropyLoss(const MatrixType& pred, const MatrixType& label, size_t numClass);

#endif // UTILS_H