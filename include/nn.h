#ifndef NN_H
#define NN_H

#include <vector>
#include <string>
#include "layer.h"
#include "utils.h"

struct LayerConfig
{
    size_t input;
    size_t output;
    std::string activation;
};

class NeuralNetwork
{
public:
    NeuralNetwork(
        const std::vector<LayerConfig>& hidden, 
        bool randomInit = true,
        const std::string& optimizer = "SGD",
        double lr = 0.001);

    MatrixPtr Forward(const MatrixPtr& input);
    void Backward(
        const std::string& lossFunc,
        const MatrixPtr& pred,
        const MatrixPtr& label
    );
    void Optimize();

    void Print();
    void ZeroGrad();

private:
    size_t m_Input;
    size_t m_Output;
    LayerVector m_Layer;
    std::string m_Optimizer;
    double m_LearningRate;
};

#endif // NN_H