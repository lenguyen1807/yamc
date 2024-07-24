#include "nn.h"
#include <functional>
#include <algorithm>

NN::NN(
    const std::vector<LayerConfig>& hidden,
    double lr,
    bool randomInit
) : m_LearningRate(lr)
{ 
    for (size_t i = 0; i < hidden.size(); i++)
    {
        size_t input = hidden[i].input;
        size_t output = hidden[i].output;
        std::string activation = hidden[i].activation;

        LayerPtr layer = std::make_unique<Layer>(input, output, activation, randomInit);
        m_Layer.emplace_back(std::move(layer));
    }

    for (size_t i = 0; i < m_Layer.size(); i++)
    {
        m_Layer[i]->Print();
    }
}

MatrixPtr NN::Forward(const Matrix& input)
{
    // calculate input with one hidden
    m_Layer[0]->Compute(input);
    MatrixPtr output = m_Layer[0]->GetOutput();

    // calculate all hidden and output
    for (size_t i = 1; i < m_Layer.size(); i++)
    {
        m_Layer[i]->Compute((*output));
        output = m_Layer[i]->GetOutput();
    }

    return output;
}