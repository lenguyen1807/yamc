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
        }
    );

    // auto trainImgs = ReadData(std::string(DATA_PATH) + "test.csv");
    auto trainImgs = ReadData(std::string(DATA_PATH) + "archive/mnist_train.csv");
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

        /* Training part */
        for (size_t i = 0; i < trainImgs.size(); i++)
        {
            // forward pass
            MatrixPtr logits = nn.Forward(trainImgs[i]->data);

            // Apply softmax
            MatrixPtr pred = SoftMax(logits);

            // calculate loss with label
            double loss = CrossEntropyLoss(*pred, *(trainImgs[i]->label));
            trainLoss += loss;

            // calculate accuracy
            size_t predLabel = pred->ArgMax();
            size_t trueLabel = trainImgs[i]->label->ArgMax();
            trainCorrect += (predLabel == trueLabel) ? 1.0 : 0.0;

            // zero all previous gradients
            nn.ZeroGrad();

            // backward pass (calculate gradients)
            nn.Backward("Cross Entropy", pred, trainImgs[i]->label);

            // optimize
            nn.Optimize();

            std::cout << "Finish train image no." << i + 1 << " with loss " << loss << "\n";
        }

        std::cout << "Epoch: " << epoch << "\n"
                  << "Train acc: " << trainCorrect / static_cast<double>(trainImgs.size()) << "\n"
                  << "Average train loss: " << trainLoss / static_cast<double>(testImgs.size()) << "\n";
        
        /* Testing part */
        for (const auto& img : testImgs)
        {
            // forward pass
            MatrixPtr logits = nn.Forward(img->data);

            // Apply softmax
            MatrixPtr pred = SoftMax(logits);

            // calculate loss with label
            testLoss += CrossEntropyLoss((*pred), *(img->label));

            // calculate accuracy
            size_t predLabel = pred->ArgMax();
            size_t trueLabel = img->label->ArgMax();
            testCorrect += (predLabel == trueLabel) ? 1.0 : 0.0;
        }

        // end time counting
        auto end = high_resolution_clock::now();
        duration<double, std::milli> time = end - start;

        std::cout << "Test acc: " << testCorrect / static_cast<double>(testImgs.size()) << "\n"
                  << "Average test loss: " << testLoss / static_cast<double>(testImgs.size()) << "\n"
                  << "Time: " << time.count() << "ms" << "\n"
                  << "-------------------------------------------" << "\n";
                  
    #ifdef _WIN32
        GetMemoryInfo();
    #endif
    }

    return 0;
}