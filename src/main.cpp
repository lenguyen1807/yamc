#include <iostream>
#include "nn.h"
#include "image.h"
// #include "matrix.h"

int main()
{
    auto imgs = Image::ReadData(std::string(DATA_PATH) + "test.csv");
    NN neuralNetwork(784, 64, 10, "Sigmoid", 0.05f);
    Matrix a = Matrix::Flatten(imgs[0]->data);
    Matrix b = neuralNetwork.FeedForward(a);
    b.Print();
    return 0;
}