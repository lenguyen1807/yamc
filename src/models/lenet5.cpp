#include "layers/activation.h"
#include "layers/avgpool.h"
#include "layers/conv.h"
#include "models/lenet5.h"

LeNet5::LeNet5(size_t input_channels, size_t output_size) {}

nn::matrix<float> LeNet5::forward(const cv::Mat& image) {}

void LeNet5::backward(const nn::matrix<float>& grad) {}
