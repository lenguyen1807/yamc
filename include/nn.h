#ifndef NN_H
#define NN_H

#include "activation.h"
#include <string>
#include <map>

// shallow neural network (with only one hidden layer)
class NN
{
public:
    NN(size_t input, size_t output, size_t hidden, std::string activation, double lr);

    Matrix FeedForward(const Matrix& input);
    void BackProp(const Matrix& output);
    void Optimize();

private:
    size_t inputSize;
    size_t outputSize;
    size_t hiddenSize;

    // activation function for layer
    Activation* activFunc;

    // activation function for output
    Activation* activOutput;

    // Store parameter (weight)
    std::map<std::string, Matrix> ws;

    // Store feed forward value
    std::map<std::string, Matrix> ff;

    // Gradient
    std::map<std::string, Matrix> grads;

    double learningRate;
};

#endif // NN_H