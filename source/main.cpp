#define GEMM_OPT

#include <chrono>

#include "data.h"
#include "loss.h"
#include "mlp.h"
#include "optimizer.h"
#include "utils.h"

int main()
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  // Reading data
  // MNISTData train_data(std::string(DATA_PATH) + "test.csv");
  MNISTData train_data(std::string(DATA_PATH) + "mnist/mnist_train.csv");
  MNISTData test_data(std::string(DATA_PATH) + "mnist/mnist_test.csv");

  // Create neural network model
  // Because softmax will calculate activation so we only need linear activation
  // at the last layer
  nn::MLP model(
      {
          {784, 500, nn::Activation::RELU},
          {500, 300, nn::Activation::RELU},
          {300, 100, nn::Activation::RELU},
          {100, 10, nn::Activation::LINEAR},
      },
      /*rand_init=*/true);

  // Create SGD optimizer
  nn::SGD optimizer(&model, /*learning_rate=*/0.001);

  // Create loss function
  nn::CrossEntropyLoss loss_fn(&model);

  // training and testing
  for (size_t epoch = 1; epoch <= EPOCHS; epoch++) {
    // start time counting
    auto start = high_resolution_clock::now();

    double train_correct {};
    double test_correct {};
    double train_loss {};
    double test_loss {};

    std::cout << "------------------ Training -----------------\n";

    size_t img_idx = 0;
    for (const auto& img : train_data.dataset) {
      // forward pass
      auto logits = model.forward(img->data);

      // calculate loss
      double loss = loss_fn(logits, img->label);
      train_loss += loss;

      // calculate accuracy
      size_t pred_label = loss_fn.get_pred().arg_max();
      size_t true_label = img->label.arg_max();
      train_correct += (pred_label == true_label) ? 1.0 : 0.0;

      // backward pass
      loss_fn.backward(img->label);

      // update weight
      optimizer.step();

      // zero all gradients for next iteration
      model.zero_grad();

      std::cout << "Finish train image no." << img_idx + 1 << " with loss "
                << loss << "\n";

      img_idx++;
    }

    std::cout << "Epoch: " << epoch << "\n"
              << "Train accuracy: "
              << train_correct / static_cast<double>(train_data.dataset.size())
              << "\n"
              << "Average train loss: "
              << train_loss / static_cast<double>(train_data.dataset.size())
              << "\n";

    // std::cout << "------------------ Testing -----------------\n";
    for (const auto& img : test_data.dataset) {
      // forward pass
      auto logits = model.forward(img->data);

      // calculate loss
      double loss = loss_fn(logits, img->label);
      test_loss += loss;

      // calculate accuracy
      size_t pred_label = loss_fn.get_pred().arg_max();
      size_t true_label = img->label.arg_max();
      test_correct += (pred_label == true_label) ? 1.0 : 0.0;
    }

    std::cout << "Test accuracy: "
              << test_correct / static_cast<double>(test_data.dataset.size())
              << "\n"
              << "Average test loss: "
              << test_loss / static_cast<double>(test_data.dataset.size())
              << "\n";

    // end time counting
    auto end = high_resolution_clock::now();
    duration<double, std::milli> time = end - start;

    std::cout << "Time: " << time.count() / 60000.0 << " minutes";
  }

  return 0;
}