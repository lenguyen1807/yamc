#include <chrono>
#include <iostream>

#include "data/MNIST.h"
#include "loss.h"
#include "models/mlp.h"
#include "optimizer.h"

int main()
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  /* -------------- Load data --------------- */
  // Load data
  MNISTData trainset(std::string(DATA_SOURCE_DIR) + "mnist/mnist_train.csv");
  std::cout << "DONE: Load training data MNIST with " << trainset.dataset.size()
            << " image\n";
  MNISTData testset(std::string(DATA_SOURCE_DIR) + "mnist/mnist_test.csv");
  std::cout << "DONE: Load test data MNIST with " << testset.dataset.size()
            << " image\n";

  /* -------------- Train model data --------------- */
  // create model, loss and optimizer
  MLP model(24 * 24, 10);
  nn::CrossEntropyLoss loss_fn(&model);
  nn::SGD optim(&model, 0.001f);

  // training and testing
  for (size_t epoch = 1; epoch <= 1; epoch++) {
    float train_correct {};
    float test_correct {};
    float train_loss {};
    float test_loss {};

    std::cout << "------------------ Training -----------------\n";

    auto start = high_resolution_clock::now();
    size_t img_idx = 0;
    model.train();

    for (const auto& img : trainset.dataset) {
      // forward pass
      auto logits = model.forward(img->data);

      // calculate loss
      float loss = loss_fn(logits, img->label);
      train_loss += loss;

      // calculate accuracy
      int pred_label = loss_fn.get_pred().arg_max();
      int true_label = img->label.arg_max();
      train_correct += (pred_label == true_label) ? 1.0f : 0.0f;

      // backward pass
      model.backward(loss_fn.get_loss_grad());

      // update weight
      optim.step();

      // zero all gradients for next iteration
      model.zero_grad();

      std::cout << "Finish train image no." << img_idx + 1 << " with loss "
                << loss << "\n";
      img_idx++;
    }

    std::cout << "Epoch: " << epoch << "\n"
              << "Train accuracy: "
              << train_correct / static_cast<float>(trainset.dataset.size())
              << "\n"
              << "Average train loss: "
              << train_loss / static_cast<float>(trainset.dataset.size())
              << "\n";

    auto end = high_resolution_clock::now();
    duration<float, std::milli> time = end - start;
    std::cout << "Traing data time: " << time.count() / 60000.0 << " minutes\n";

    std::cout << "------------------ Testing -----------------\n";

    start = high_resolution_clock::now();
    model.eval();

    for (const auto& img : testset.dataset) {
      // forward pass
      auto logits = model.forward(img->data);

      // calculate loss
      float loss = loss_fn(logits, img->label);
      test_loss += loss;

      // calculate accuracy
      int pred_label = loss_fn.get_pred().arg_max();
      int true_label = img->label.arg_max();
      test_correct += (pred_label == true_label) ? 1.0f : 0.0f;
    }

    std::cout << "Test accuracy: "
              << test_correct / static_cast<float>(testset.dataset.size())
              << "\n"
              << "Average test loss: "
              << test_loss / static_cast<float>(testset.dataset.size()) << "\n";

    end = high_resolution_clock::now();
    time = end - start;
    std::cout << "Testing data time: " << time.count() / 60000.0
              << " minutes\n";
  }

  return 0;
}
