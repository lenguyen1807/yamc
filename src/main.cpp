#include <chrono>

#include "activation.h"
#include "dropout.h"
#include "linear.h"
#include "loss.h"
#include "matrix.h"
#include "module.h"
#include "optimizer.h"

int main()
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  // Create model
  nn::Module model {};
  model.add<nn::Linear>(784, 500);
  model.add<nn::Dropout>();
  model.add<nn::ReLU>();
  model.add<nn::Linear>(500, 300);
  model.add<nn::Dropout>();
  model.add<nn::ReLU>();
  model.add<nn::Linear>(300, 100);
  model.add<nn::Dropout>();
  model.add<nn::ReLU>();
  model.add<nn::Linear>(100, 10);

  // Create optimizer
  nn::SGD optim(&model, 0.01f);
  // Create loss
  nn::CrossEntropyLoss loss_fn(&model);

  // start time counting
  auto start = high_resolution_clock::now();
  nn::matrix<float> x = nn::matrix<float>::nrand(784, 1, 0.0f, 1.0f);

  // forward model
  auto logits = model.forward(x);

  // end time counting
  auto end = high_resolution_clock::now();
  duration<float, std::milli> time = end - start;
  std::cout << "Time: " << time.count() / 60000.0 << " minutes\n";

  return 0;
}