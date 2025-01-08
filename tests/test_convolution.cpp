#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>

#include "layers/conv.h"
#include "matrix.h"
#include "test_helper.h"

TEST_CASE("im2col on 1x5x5 image, stride = 2, padding = 0, kernel = 3",
          "[im2col-1x5x5-2-0-3]")
{
  cv::Mat img(5, 5, CV_32F);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++)
      img.at<float>(i, j) = i + j;

  nn::matrix<float> base({{0, 2, 2, 4},
                          {1, 3, 3, 5},
                          {2, 4, 4, 6},
                          {1, 3, 3, 5},
                          {2, 4, 4, 6},
                          {3, 5, 5, 7},
                          {2, 4, 4, 6},
                          {3, 5, 5, 7},
                          {4, 6, 6, 8}});
  nn::matrix<float> col_mat = nn::Convolution::im2col(img, 3, 3, 2, 2, 0, 0);
  REQUIRE(is_close_mat(base, col_mat, EPSILON_FLT));
}

// TODO: Implement more test cases to ensure im2col and col2im function work
// correctly (hope so)