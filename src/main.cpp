#include <iostream>
#include <chrono>

#include "nn.h"
#include "image.h"
#include "utils.h"

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    NeuralNetwork nn(
        {
            {784, 500, "ReLU"}, 
            {500, 300, "ReLU"},
            {300, 100, "ReLU"},
            {100, 10, "Linear"}
        }, 
        true
    );

    // auto imgs = ReadData(std::string(DATA_PATH) + "archive/mnist_test.csv");
    auto trainImgs = ReadData(std::string(DATA_PATH) + "test.csv");
    auto testImgs = ReadData(std::string(DATA_PATH) + "archive/mnist_test.csv");

    // training and testing
    for (size_t epoch = 1; epoch <= EPOCHS; epoch++)
    {
        // start time counting
        auto start = high_resolution_clock::now();

        double trainCorrect;
        double testCorrect;
        double trainLoss;
        double testLoss;

        for (const auto& img : trainImgs)
        {
            /* Training part */

            // forward pass
            MatrixPtr logits = nn.Forward(img->data);

            // Apply softmax
            Matrix pred = SoftMax(*logits);

            // calculate loss with label
            trainLoss += CrossEntropyLoss(pred, *(img->label));

            // calculate accuracy
            size_t predLabel = pred.ArgMax();
            size_t trueLabel = img->label->ArgMax();
            trainCorrect += (predLabel == trueLabel) ? 1.0 : 0.0;

            // zero all previous gradients
            nn.ZeroGrad();

            // backward pass (calculate gradients)

            // optimize
        }

        std::cout << "Epoch: " << epoch << "\n"
                  << "Train acc: " << trainCorrect / static_cast<double>(trainImgs.size()) << "\n"
                  << "Average train loss: " << trainLoss / static_cast<double>(trainImgs.size()) << "\n";
        
        for (const auto& img : testImgs)
        {
            /* Testing part */

            // enable testing mode
            nn.trainMode = false;

            // forward pass
            MatrixPtr logits = nn.Forward(img->data);

            // Apply softmax
            Matrix pred = SoftMax(*logits);

            // calculate loss with label
            testLoss += CrossEntropyLoss(pred, *(img->label));

            // calculate accuracy
            size_t predLabel = pred.ArgMax();
            size_t trueLabel = img->label->ArgMax();
            testCorrect += (predLabel == trueLabel) ? 1.0 : 0.0;
        }

        // end time counting
        auto end = high_resolution_clock::now();

        duration<double, std::milli> time = end - start;

        std::cout << "Test acc: " << testCorrect / static_cast<double>(testImgs.size()) << "\n"
                  << "Average test loss: " << testLoss / static_cast<double>(testImgs.size()) << "\n"
                  << "Time: " << time.count() << "\n"
                  << "-------------------------------------------" << "\n";
    }

    return 0;
}