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

class NN
{
public:
    NN(const std::vector<LayerConfig>& hidden, double lr, bool randomInit);
    MatrixPtr Forward(const Matrix& input);

private:
    size_t m_Input;
    size_t m_Output;
    LayerVector m_Layer;
    double m_LearningRate;
};

#endif // NN_H