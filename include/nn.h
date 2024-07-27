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

struct TrainConfig
{
    double lr;
    size_t epochs;
    size_t batches = 16;
    bool batch = false;
    std::string optimizer;
};

class NeuralNetwork
{
public:
    bool trainMode;

    NeuralNetwork(const std::vector<LayerConfig>& hidden, bool randomInit);

    MatrixPtr Forward(const MatrixPtr& input);

    void Print();
    void ZeroGrad();

private:
    size_t m_Input;
    size_t m_Output;
    LayerVector m_Layer;
    double m_LearningRate;
};

#endif // NN_H