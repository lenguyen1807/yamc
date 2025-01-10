#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>

#include "layers/avgpool.h"
#include "layers/conv.h"

int main()
{
  cv::Mat img = cv::imread(std::string(DATA_SOURCE_DIR)
                               + "cifar-10/test/abandoned_ship_s_000213.png",
                           cv::IMREAD_COLOR);
  cv::Mat img2;
  img.convertTo(img2, CV_32FC3);

  nn::AvgPool2D pool(3, 1, 0);
  nn::Conv2D conv(/*input_channels=*/3,
                  /*output_channels=*/2,
                  /*stride=*/1,
                  /*kernel_size=*/3,
                  /*padding=*/1);

  cv::Mat result = conv.forward(img2);
  std::cout << result << "\n";

  // cv::Mat result1 = pool.forward(result);
  // std::cout << result1 << "\n";

  return 0;
}
