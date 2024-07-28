#include "nn.h"
#include <functional>
#include <algorithm>

NeuralNetwork::NeuralNetwork(
    const std::vector<LayerConfig>& hidden,
    bool randomInit,
    const std::string& optimizer,
    double lr
) : m_LearningRate(lr)
{ 
    // get optimizer
    m_Optimizer = optimizer;
    std::transform(m_Optimizer.begin(), m_Optimizer.end(), m_Optimizer.begin(), ::tolower);

    // get layer
    for (size_t i = 0; i < hidden.size(); i++)
    {
        size_t input = hidden[i].input;
        size_t output = hidden[i].output;
        std::string activation = hidden[i].activation;

        LayerPtr layer = std::make_unique<Layer>(input, output, activation, randomInit);
        m_Layer.emplace_back(std::move(layer));
    }
}

MatrixPtr NeuralNetwork::Forward(const MatrixPtr& input)
{
    // calculate input with one hidden
    m_Layer[0]->Compute(input);
    MatrixPtr output = m_Layer[0]->GetOutput();

    // calculate all hidden and output
    for (size_t i = 1; i < m_Layer.size(); i++)
    {
        m_Layer[i]->Compute(output);
        output = m_Layer[i]->GetOutput();
    }

    return output;
}

void NeuralNetwork::Print()
{
    for (size_t i = 0; i < m_Layer.size(); i++)
    {
        std::cout << "Layer " << i + 1 << ":\n";
        m_Layer[i]->Print();
    }
    std::cout << "Optimizer: " << m_Optimizer << "\n";
    std::cout << "Learning rate: " << m_LearningRate << "\n";
}

void NeuralNetwork::ZeroGrad()
{
    for (size_t i = 0; i < m_Layer.size(); i++)
    {
        m_Layer[i]->ZeroGradient();
    }
}

void NeuralNetwork::Backward(
    const std::string& lossFunc,
    const MatrixPtr& pred,
    const MatrixPtr& label
)
{
    std::string loss = lossFunc;
    std::transform(loss.begin(), loss.end(), loss.begin(), ::tolower);

    // calculate gradient of loss function
    MatrixPtr lossGrad;
    if (loss == "cross entropy")
    {
        // assume the pred is from softmax activation
        lossGrad = std::make_shared<Matrix>((*pred) - (*label));
    }
    else
    {
        throw "Not implemented yet :>";
    }

    // calculate gradient of the output
    size_t lastIdx = m_Layer.size() - 1;

    m_Layer[lastIdx]->Gradient(lossGrad, m_Layer[lastIdx - 1]->GetOutput());
    MatrixPtr grad = m_Layer[lastIdx]->GetGradient();

    // calculate gradient of other layers
    for (size_t i = lastIdx; i-- > 1;)
    {
        m_Layer[i]->Gradient(grad, m_Layer[i - 1]->GetOutput());
        grad = m_Layer[i]->GetGradient();
    }

    // delete lossGrad matrix (we don't need it anymore)
    lossGrad.reset();
}

void NeuralNetwork::Optimize()
{
    // update weight for each layer based on its gradient
    if (m_Optimizer == "sgd")
    {
        for (size_t i = 0; i < m_Layer.size(); i++)
        {
            MatrixPtr weight = m_Layer[i]->GetWeight();
            MatrixPtr grad = m_Layer[i]->GetWeightGrad();
            (*weight) = (*weight) - (m_LearningRate * (*grad));
        }
    }
    else
    {
        throw "Not implemented yet :>";
    }
}