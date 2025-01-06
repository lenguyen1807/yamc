#include <chrono>
#include <iostream>

#include "data/CIFAR10.h"

int main()
{
  // Time checking
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  /* -------------- Load data --------------- */
  auto start = high_resolution_clock::now();

  CIFAR10Data dataset {};

  auto end = high_resolution_clock::now();
  duration<float, std::milli> time = end - start;
  std::cout << "Load data time: " << time.count() / 60000.0 << " minutes\n";

  /* -------------- Train model data --------------- */

  return 0;
}
