#include <chrono>
#include <iostream>

#include "data/TinyImageNet.h"
#include "loss.h"
#include "models/alexnet.h"

int main()
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  /* -------------- Load data --------------- */
  auto start = high_resolution_clock::now();

  TinyImageNetData test(false);

  auto end = high_resolution_clock::now();
  duration<float, std::milli> time = end - start;
  std::cout << "Load data time: " << time.count() / 60000.0 << " minutes\n";

  /* -------------- Test model --------------- */

  AlexNet model(3, 200);
  nn::CrossEntropyLoss loss_fn(&model);

  model.eval();
  float test_loss {};
  float test_correct {};

  for (const auto& img : test.dataset) {
    // forward pass
    auto logits = model.forward(img->data);

    // calculate loss
    float loss = loss_fn(logits, img->label);
    test_loss += loss;

    // calculate accuracy
    int pred_label = loss_fn.get_pred().arg_max();
    int true_label = img->label.arg_max();
    test_correct += (pred_label == true_label) ? 1.0f : 0.0f;

    std::cout << "Finish test image with loss: " << loss << "\n";
  }

  std::cout << "Test accuracy: "
            << test_correct / static_cast<float>(test.dataset.size()) << "\n"
            << "Average test loss: "
            << test_loss / static_cast<float>(test.dataset.size()) << "\n";

  return 0;
}
