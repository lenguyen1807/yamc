#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>

#include "layers/conv.h"

int main()
{
  cv::Mat img(5, 5, CV_32F);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++)
      img.at<float>(i, j) = i + j;

  nn::Convolution conv(/*input_channels=*/1,
                       /*output_channels=*/3,
                       /*stride=*/2,
                       /*padding=*/0,
                       /*kernel_size=*/3);

  cv::Mat result = conv.forward(img);

  return 0;
}
