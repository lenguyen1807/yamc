#include <chrono>

#include "data.h"
#include "mlp.h"
#include "utils.h"

auto main() -> int
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  // Reading data
  MNISTData train_data(std::string(DATA_PATH) + "test.csv");
  // MNISTData train_data(std::string(DATA_PATH) + "archive/mnist_train.csv");
  // MNISTData test_data(std::string(DATA_PATH) + "archive/mnist_test.csv");

  // Create neural network model
  // Because softmax will calculate activation so we only need linear activation
  // at the last layer
  nn::MLP model(
      {
          {784, 128, nn::Activation::RELU},
          {128, 10, nn::Activation::LINEAR},
      },
      /*rand_init=*/true);

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
      nn::matrix<double> logits = model.forward(img->data);

      // apply softmax
      nn::matrix<double> pred = nn::softmax(logits);

      // calculate loss
      double loss = nn::cross_entropy_loss(pred, img->label);
      train_loss += loss;

      // calculate accuracy
      size_t pred_label = pred.arg_max();
      size_t true_label = img->label.arg_max();
      train_correct += (pred_label == true_label) ? 1.0 : 0.0;

      // backward pass
      model.backward(pred, img->label);

      // optimize
      model.optimize();

      // zero all gradients for next iteration
      model.zero_grad();

      std::cout << "Finish train image no." << img_idx + 1 << " with loss "
                << loss << "\n";
      img_idx++;
    }

    std::cout << "Epoch: " << epoch << "\n"
              << "Train acc: "
              << train_correct / static_cast<double>(train_data.dataset.size())
              << "\n"
              << "Average train loss: "
              << train_loss / static_cast<double>(train_data.dataset.size())
              << "\n";

    // end time counting
    auto end = high_resolution_clock::now();
    duration<double, std::milli> time = end - start;

    std::cout << "Time: " << time.count() << "ms";
  }

  return 0;
}
