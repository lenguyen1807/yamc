#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <vector>
#include <cmath>

constexpr size_t EPOCHS = 1;

using MatrixPtr = std::shared_ptr<class Matrix>;
using ImagePtr = std::unique_ptr<class Image>;
using LayerPtr = std::unique_ptr<class Layer>;
using MatrixVector = std::vector<MatrixPtr>;
using ImageVector = std::vector<ImagePtr>;
using LayerVector = std::vector<LayerPtr>;

double Sigmoid(double x);

double ReLU(double x);

double Linear(double x);

class Matrix SoftMax(const class Matrix& x);

double CrossEntropyLoss(const class Matrix& pred, const class Matrix& label);

#endif // UTILS_H