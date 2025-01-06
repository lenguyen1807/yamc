#include "models/module.h"

using namespace nn;

void Module::zero_grad()
{
  for (auto it = layers.begin(); it != layers.end(); ++it) {
    it->second->zero_grad();
  }
}

void Module::train()
{
  for (auto it = layers.begin(); it != layers.end(); ++it) {
    it->second->train = true;
  }
}

void Module::eval()
{
  for (auto it = layers.begin(); it != layers.end(); ++it) {
    it->second->train = false;
  }
}