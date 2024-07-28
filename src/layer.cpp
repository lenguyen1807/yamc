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
, m_Gradient(std::make_shared<Matrix>(inputSize, 1))
, m_WeightGrad(std::make_shared<Matrix>(outputSize, inputSize))
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

    m_ActivationName = result;

    // create activation function
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

// compute forward pass
void Layer::Compute(const MatrixPtr& input)
{
    *m_PreActiv = (*m_Weight) * (*input);
    *m_AfterActiv = (*m_PreActiv).Apply(m_Activation);
}

// compute gradient
void Layer::Gradient(const MatrixPtr& prevGrad, const MatrixPtr& prevLayer)
{
    // step 1: compute layer after activation
    std::function<double(double)> gradFunc;
    if (m_ActivationName == "sigmoid")      gradFunc = SigmoidGrad;
    else if (m_ActivationName == "relu")    gradFunc = ReLUGrad;
    else if (m_ActivationName == "linear")  gradFunc = LinearGrad;
    else                                    gradFunc = LinearGrad;

    Matrix grad = (*prevGrad) % (m_PreActiv->Apply(gradFunc));

    // step 2: compute gradient respect to weight
    (*m_WeightGrad) = grad * (prevLayer->T());

    // step 3: compute pre-activation layer
    (*m_Gradient) = (m_Weight->T()) * grad;
}

void Layer::ZeroGradient()
{
    m_Gradient->Fill(0.0);
}

MatrixPtr Layer::GetOutput() const
{
    return m_AfterActiv;
}

MatrixPtr Layer::GetGradient() const
{
    return m_Gradient;
}

MatrixPtr Layer::GetWeight() const
{
    return m_Weight;
}

MatrixPtr Layer::GetWeightGrad() const
{
    return m_WeightGrad;
}