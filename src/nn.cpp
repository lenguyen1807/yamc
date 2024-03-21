#include "nn.h"

NN::NN(size_t input, size_t hidden, size_t output, std::string activation, double lr)
: inputSize(input)
, outputSize(output)
, hiddenSize(hidden)
, learningRate(lr)
{
    // Set activation
    if (activation == "ReLU")         { activFunc = new ReLU(); }
    else if (activation == "Sigmoid") { activFunc = new Sigmoid(); }

    // Set weights
    ws.emplace("input_hidden", Matrix::Randomized(hiddenSize, inputSize));
    ws.emplace("bias_hidden", Matrix(hiddenSize, 1)); // initialized bias as 0 matrix
    ws.emplace("hidden_output", Matrix::Randomized(outputSize, hiddenSize));
    ws.emplace("bias_output", Matrix(outputSize, 1));
}

Matrix NN::FeedForward(const Matrix& input)
{
    // Input layer value
    ff.emplace("input", input);

    // Hidden layer value
    ff.emplace("hidden", (ws.at("input_hidden") * ff.at("input")) + ws.at("bias_hidden"));
    ff.emplace("hidden_activ", activFunc->Apply(ff.at("hidden")));

    // Output layer value
    ff.emplace("output", (ws.at("hidden_output") * ff.at("hidden_activ")) + ws.at("bias_output"));
    ff.emplace("output_activ", Activation::SoftMax(ff.at("output")));

    return ff.at("output_activ");
}

void NN::BackProp(const Matrix& output)
{
}

void NN::Optimize()
{
    
}