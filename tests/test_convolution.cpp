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
  nn::matrix<float> col_mat = nn::Conv2D::im2col(img, 3, 3, 2, 2, 0, 0);
  REQUIRE(is_close_mat(base, col_mat, EPSILON_FLT));
}

TEST_CASE("col2im on 1x5x5 image, stride = 2, padding = 0, kernel = 3",
          "[col2im-1x5x5-2-0-3]")
{
  // NOTE THAT: col2im don't have to reverse that im2col did
  // https://stackoverflow.com/questions/51703367/col2im-implementation-in-convnet
}

// TODO: Implement more test cases to ensure im2col and col2im function work
// correctly (hope so)

TEST_CASE("reshape matrix to image", "reshape_mat2im")
{
  cv::Mat img(5, 5, CV_32F);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++)
      img.at<float>(i, j) = i + j;

  nn::matrix<float> col_mat = nn::Conv2D::reshape_im2mat(img);
  REQUIRE(is_close_im(nn::Conv2D::reshape_mat2im(col_mat, 1, 5, 5), img));
}

TEST_CASE("reshape image to matrix", "reshape_im2mat")
{
  cv::Mat img(5, 5, CV_32F);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++)
      img.at<float>(i, j) = i + j;

  nn::matrix<float> base({{0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4,
                           5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8}});
  nn::matrix<float> col_mat = nn::Conv2D::reshape_im2mat(img);
  REQUIRE(is_close_mat(base, col_mat, EPSILON_FLT));
}