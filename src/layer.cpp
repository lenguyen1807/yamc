#include "layer.h"
#include <algorithm>

Layer::Layer(
    size_t inputSize, 
    size_t outputSize, 
    const std::string& activation,
    bool randomInit
)
: m_PreActiv(nullptr)
, m_AfterActiv(nullptr)
{
    // create weight matrix
    if (randomInit)
        m_Weight = std::make_shared<Matrix>(Matrix::Randomized(outputSize, inputSize));
    else
        m_Weight = std::make_shared<Matrix>(outputSize, inputSize);

    // lowercase a string
    // https://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
    std::string result = activation;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    if (result == "sigmoid")        m_Activation = Sigmoid;
    else if (result == "relu")      m_Activation = ReLU;
    else if (result == "linear")    m_Activation = Linear;
    else                            m_Activation = Linear;
}

void Layer::Print()
{
    std::cout << "Input: " << m_Weight->cols << ", "
              << "Ouput: " << m_Weight->rows << "\n";
}

void Layer::Compute(const Matrix& input)
{
    m_PreActiv = std::make_shared<Matrix>((*m_Weight) * input);
    m_AfterActiv = std::make_shared<Matrix>((*m_PreActiv).Apply(m_Activation));
}

MatrixPtr Layer::GetOutput() const
{
    return m_AfterActiv;
}