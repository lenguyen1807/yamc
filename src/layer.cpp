#include "layer.h"
#include <algorithm>

Layer::Layer(
    size_t inputSize, 
    size_t outputSize, 
    const std::string& activation,
    bool randomInit
)
: m_PreActiv(std::make_shared<Matrix>(outputSize, 1))
, m_AfterActiv(std::make_shared<Matrix>(outputSize, 1))
, m_Gradient(std::make_shared<Matrix>(outputSize, 1))
{
    // create weight matrix
    if (randomInit)
        m_Weight = std::make_shared<Matrix>(Matrix::Randomized(outputSize, inputSize));
    else
        m_Weight = std::make_shared<Matrix>(outputSize, inputSize);

    // lowercase a string
    // https://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
    std::string result = activation;

    m_ActivationName = result;

    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    if (result == "sigmoid")        m_Activation = Sigmoid;
    else if (result == "relu")      m_Activation = ReLU;
    else if (result == "linear")    m_Activation = Linear;
    else                            m_Activation = Linear;
}

void Layer::Print()
{
    std::cout << "Input: " << m_Weight->cols << ", "
              << "Ouput: " << m_Weight->rows << ", "
              << "Activation: " << m_ActivationName << "\n";
}

void Layer::Compute(const MatrixPtr& input)
{
    *m_PreActiv = (*m_Weight) * (*input);
    *m_AfterActiv = (*m_PreActiv).Apply(m_Activation);
}

void Layer::ZeroGradient()
{
    (*m_Gradient).Fill(0.0);
}

MatrixPtr Layer::GetOutput() const
{
    return m_AfterActiv;
}