#include "layers/activation.h"
#include "layers/conv.h"
#include "layers/dropout.h"
#include "layers/flatten.h"
#include "layers/linear.h"
#include "layers/lrn.h"
#include "layers/maxpool.h"
#include "models/alexnet.h"

// NOTE: The input of AlexNet image should 227 x 227 as the original paper

AlexNet::AlexNet(size_t input_channels, size_t output_size)
{
  // 3 x 227 x 227
  add<0, nn::Conv2D>(input_channels,
                     /*output_channels=*/96,
                     /*stride=*/4,
                     /*kernel_size=*/11,
                     /*padding=*/0);
  // 96 x 55 x 55
  add<1, nn::ReLU>();
  // 96 x 55 x 55
  add<2, nn::LocalResponseNorm>();
  // 96 x 55 x 55
  add<3, nn::Maxpool2D>(/*kernel_size=*/3,
                        /*stride=*/2,
                        /*padding=*/0);

  // 96 x 27 x 27
  add<4, nn::Conv2D>(/*input_channels=*/96,
                     /*output_channels=*/256,
                     /*stride=*/1,
                     /*kernel_size=*/5,
                     /*padding=*/2);
  // 256 x 27 x 27
  add<5, nn::ReLU>();
  // 256 x 27 x 27
  add<6, nn::LocalResponseNorm>();
  // 256 x 27 x 27
  add<7, nn::Maxpool2D>(/*kernel_size=*/3,
                        /*stride=*/2,
                        /*padding=*/0);

  // 256 x 13 x 13
  add<8, nn::Conv2D>(/*input_channels=*/256,
                     /*output_channels=*/384,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  // 384 x 13 x 13
  add<9, nn::ReLU>();
  // 384 x 13 x 13
  add<10, nn::Conv2D>(/*input_channels=*/384,
                      /*output_channels=*/384,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  // 384 x 13 x 13
  add<11, nn::ReLU>();
  // 384 x 13 x 13
  add<12, nn::Conv2D>(/*input_channels=*/384,
                      /*output_channels=*/256,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  // 256 x 13 x 13
  add<13, nn::ReLU>();
  // 256 x 13 x 13
  add<14, nn::Maxpool2D>(/*kernel_size=*/3,
                         /*stride=*/2,
                         /*padding=*/0);

  // 256 x 6 x 6
  add<15, nn::Flatten>();
  // 256 * 6 * 6
  add<16, nn::Dropout>();
  add<17, nn::Linear>(256 * 6 * 6, 4096);
  add<18, nn::ReLU>();
  add<19, nn::Dropout>();
  add<20, nn::Linear>(4096, 4096);
  add<21, nn::ReLU>();
  add<22, nn::Linear>(4096, output_size);
}

nn::matrix<float> AlexNet::forward(const cv::Mat& input)
{
  // TODO
}

void AlexNet::backward(const nn::matrix<float>& grad)
{
  // TODO
}