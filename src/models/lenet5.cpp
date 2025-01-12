#include "layers/activation.h"
#include "layers/avgpool.h"
#include "layers/conv.h"
#include "layers/dropout.h"
#include "layers/flatten.h"
#include "layers/linear.h"
#include "models/lenet5.h"

LeNet5::LeNet5(size_t input_channels, size_t output_size)
{
  add<0, nn::Conv2D>(input_channels,
                     /*output_channels=*/6,
                     /*stride=*/1,
                     /*kernel_size=*/5,
                     /*padding=*/0);
  add<1, nn::ReLU>();
  add<2, nn::AvgPool2D>(/*kernel_size=*/2,
                        /*stride=*/2,
                        /*padding=*/0);
  add<3, nn::Conv2D>(/*input_channels=*/6,
                     /*output_channels=*/16,
                     /*stride=*/1,
                     /*kernel_size=*/5,
                     /*padding=*/0);
  add<4, nn::ReLU>();
  add<5, nn::AvgPool2D>(/*kernel_size=*/2,
                        /*stride=*/2,
                        /*padding=*/0);
  add<6, nn::Flatten>();
  add<7, nn::Linear>(400, 84);
  add<8, nn::ReLU>();
  add<9, nn::Dropout>();  // avoid overfitting
  add<10, nn::Linear>(84, output_size);
}

nn::matrix<float> LeNet5::forward(const cv::Mat& image)
{
  // Feature extraction
  cv::Mat im = layers.at(0)->forward(image);  // Conv2D
  im = layers.at(1)->forward(im);  // ReLU
  im = layers.at(2)->forward(im);  // AvgPool2D
  im = layers.at(3)->forward(im);  // Conv2D
  im = layers.at(4)->forward(im);  // ReLU
  im = layers.at(5)->forward(im);  // AvgPool2D

  // Flatten an image to 1D vector
  nn::matrix<float> result =
      dynamic_cast<nn::Flatten*>(layers.at(6).get())->forward_im(im);

  // Classifier
  result = layers.at(7)->forward(result);  // Linear
  result = layers.at(8)->forward(result);  // ReLU
  result = layers.at(9)->forward(result);  // Dropout
  result = layers.at(10)->forward(result);  // Linear
  return result;
}

void LeNet5::backward(const nn::matrix<float>& grad)
{
  nn::matrix<float> result = layers.at(10)->backward(grad);  // Linear
  result = layers.at(9)->backward(result);  // Dropout
  result = layers.at(8)->backward(result);  // ReLU
  result = layers.at(7)->backward(result);  // Linear

  // backward 1D vector to image
  cv::Mat im =
      dynamic_cast<nn::Flatten*>(layers.at(6).get())->backward_im(result);

  im = layers.at(5)->backward(im);  // AvgPool2D
  im = layers.at(4)->backward(im);  // ReLU
  im = layers.at(3)->backward(im);  // Conv2D
  im = layers.at(2)->backward(im);  // AvgPool2D
  im = layers.at(1)->backward(im);  // ReLU
  im = layers.at(0)->backward(im);  // Conv2D
}
