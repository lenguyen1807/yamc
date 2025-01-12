#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>

#include "layers/flatten.h"
#include "matrix.h"
#include "test_helper.h"

TEST_CASE("Flatten 1x3x3 image", "[flatten-1x3x3]")
{
  // TODO:
  cv::Mat img(3, 3, CV_32F);
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      img.at<float>(i, j) = i + j;

  nn::matrix<float> base({
      {0},
      {1},
      {2},
      {1},
      {2},
      {3},
      {2},
      {3},
      {4},
  });

  nn::Flatten layer {};
  auto result = layer.forward_im(img);
  REQUIRE(is_close_mat(result, base, EPSILON_FLT));
}

TEST_CASE("Backward Flatten 1x5x5 image", "[backward-flatten-1x5x5]")
{
  cv::Mat img(3, 3, CV_32F);
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      img.at<float>(i, j) = i + j;

  nn::Flatten layer {};
  auto result = layer.backward_im(layer.forward_im(img));

  REQUIRE(is_close_im(img, result));
}