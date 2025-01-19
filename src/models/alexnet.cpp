#include "layers/activation.h"
#include "layers/conv.h"
#include "layers/dropout.h"
#include "layers/flatten.h"
#include "layers/linear.h"
#include "layers/lrn.h"
#include "layers/maxpool.h"
#include "models/alexnet.h"

AlexNet::AlexNet(size_t input_channels, size_t output_size)
{
  /*
  -------------------------------------------------------------
        Layer (type)               Output Shape         Param #
  =============================================================
            Conv2d-1           [-1, 64, 55, 55]          23,296
              ReLU-2           [-1, 64, 55, 55]               0
         MaxPool2d-3           [-1, 64, 27, 27]               0
            Conv2d-4          [-1, 192, 27, 27]         307,392
              ReLU-5          [-1, 192, 27, 27]               0
         MaxPool2d-6          [-1, 192, 13, 13]               0
            Conv2d-7          [-1, 384, 13, 13]         663,936
              ReLU-8          [-1, 384, 13, 13]               0
            Conv2d-9          [-1, 256, 13, 13]         884,992
             ReLU-10          [-1, 256, 13, 13]               0
           Conv2d-11          [-1, 256, 13, 13]         590,080
             ReLU-12          [-1, 256, 13, 13]               0
        MaxPool2d-13            [-1, 256, 6, 6]               0
          Dropout-15                 [-1, 9216]               0
           Linear-16                 [-1, 4096]      37,752,832
             ReLU-17                 [-1, 4096]               0
          Dropout-18                 [-1, 4096]               0
           Linear-19                 [-1, 4096]      16,781,312
             ReLU-20                 [-1, 4096]               0
           Linear-21                 [-1, 1000]       4,097,000
  */

  add<0, nn::Conv2D>(input_channels,
                     /*output_channels=*/64,
                     /*stride=*/4,
                     /*kernel_size=*/11,
                     /*padding=*/2);
  add<1, nn::ReLU>();
  add<2, nn::LocalResponseNorm>();
  add<3, nn::Maxpool2D>(/*kernel_size=*/3,
                        /*stride=*/2,
                        /*padding=*/0);
  add<4, nn::Conv2D>(/*input_channels=*/64,
                     /*output_channels=*/192,
                     /*stride=*/1,
                     /*kernel_size=*/5,
                     /*padding=*/2);
  add<5, nn::ReLU>();
  add<6, nn::LocalResponseNorm>();
  add<7, nn::Maxpool2D>(/*kernel_size=*/3,
                        /*stride=*/2,
                        /*padding=*/0);
  add<8, nn::Conv2D>(/*input_channels=*/192,
                     /*output_channels=*/384,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  add<9, nn::ReLU>();
  add<10, nn::Conv2D>(/*input_channels=*/384,
                      /*output_channels=*/256,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  add<11, nn::ReLU>();
  add<12, nn::Conv2D>(/*input_channels=*/256,
                      /*output_channels=*/256,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  add<13, nn::ReLU>();
  add<14, nn::Maxpool2D>(/*kernel_size=*/3,
                         /*stride=*/2,
                         /*padding=*/0);

  add<15, nn::Flatten>();
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
  cv::Mat im = layers.at(0)->forward(input);  // Conv2D
  im = layers.at(1)->forward(im);  // ReLU
  im = layers.at(2)->forward(im);  // Local Response Norm
  im = layers.at(3)->forward(im);  // Maxpool2D

  im = layers.at(4)->forward(im);  // Conv2D
  im = layers.at(5)->forward(im);  // ReLU
  im = layers.at(6)->forward(im);  // Local Response Norm
  im = layers.at(7)->forward(im);  // Maxpool2D

  im = layers.at(8)->forward(im);  // Conv2D
  im = layers.at(9)->forward(im);  // ReLU
  im = layers.at(10)->forward(im);  // Conv2D
  im = layers.at(11)->forward(im);  // ReLU
  im = layers.at(12)->forward(im);  // Conv2D
  im = layers.at(13)->forward(im);  // ReLU
  im = layers.at(14)->forward(im);  // Maxpool2D

  nn::matrix<float> result =
      dynamic_cast<nn::Flatten*>(layers.at(15).get())->forward_im(im);

  result = layers.at(16)->forward(result);  // Dropout
  result = layers.at(17)->forward(result);  // Linear
  result = layers.at(18)->forward(result);  // ReLU
  result = layers.at(19)->forward(result);  // Dropout
  result = layers.at(20)->forward(result);  // Linear
  result = layers.at(21)->forward(result);  // ReLU
  result = layers.at(22)->forward(result);  // Linear

  return result;
}

void AlexNet::backward(const nn::matrix<float>& grad)
{
  nn::matrix<float> result = layers.at(22)->backward(grad);
  result = layers.at(21)->backward(result);
  result = layers.at(20)->backward(result);
  result = layers.at(19)->backward(result);
  result = layers.at(18)->backward(result);
  result = layers.at(17)->backward(result);
  result = layers.at(16)->backward(result);

  /* Backward Flatten */
  cv::Mat im =
      dynamic_cast<nn::Flatten*>(layers.at(15).get())->backward_im(result);

  im = layers.at(14)->backward(im);  // Maxpool2D
  im = layers.at(13)->backward(im);  // ReLU
  im = layers.at(12)->backward(im);  // Conv2D
  im = layers.at(11)->backward(im);  // ReLU
  im = layers.at(10)->backward(im);  // Conv2D
  im = layers.at(9)->backward(im);  // ReLU
  im = layers.at(8)->backward(im);  // Conv2D

  im = layers.at(7)->backward(im);  // Maxpool2D
  im = layers.at(6)->backward(im);  // Local Response Norm
  im = layers.at(5)->backward(im);  // ReLU
  im = layers.at(4)->backward(im);  // Conv2D

  im = layers.at(3)->backward(im);  // Maxpool2D
  im = layers.at(2)->backward(im);  // Local Respone Norm
  im = layers.at(1)->backward(im);  // ReLU
  im = layers.at(0)->backward(im);  // Conv2D
}