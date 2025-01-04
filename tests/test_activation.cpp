#include <catch2/catch_test_macros.hpp>

#include "layers/activation.h"
#include "matrix.h"
#include "test_helper.h"

TEST_CASE("ReLU", "[relu]")
{
  nn::matrix<float> input({{-1.f, -2.f, 3.f}, {4.f, 5.f, -6.f}});
  nn::matrix<float> base({{0.f, 0.f, 3.f}, {4.f, 5.f, 0.f}});
  nn::ReLU relu;
  REQUIRE(is_close_mat(relu.forward(input), base, EPSILON_FLT));
}

TEST_CASE("ReLU backward", "[back_relu]")
{
  nn::matrix<float> input({{-1.f, -2.f, 3.f}, {4.f, 5.f, -6.f}});
  nn::matrix<float> base({{0.f, 0.f, 3.f}, {4.f, 5.f, 0.f}});
  nn::ReLU relu;
  auto result = relu.forward(input);
  REQUIRE(is_close_mat(relu.backward(input), base, EPSILON_FLT));
}

TEST_CASE("Softmax", "[softmax]")
{
  nn::matrix<float> input({{3.0f}, {1.0f}, {0.2f}});
  nn::matrix<float> base({{0.8360188f}, {0.11314284f}, {0.05083836f}});
  nn::Softmax softmax;
  REQUIRE(is_close_mat(softmax.forward(input), base, EPSILON_FLT));
}