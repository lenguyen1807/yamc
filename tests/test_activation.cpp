#include <catch2/catch_test_macros.hpp>

#include "layers/activation.h"
#include "matrix.h"
#include "test_helper.h"

TEST_CASE("ReLU", "[relu]")
{
  nn::matrix<float> input({
      {-1.f, -2.f, 3.f},
      {4.f, 5.f, -6.f},
  });

  nn::matrix<float> base({
      {0.f, 0.f, 3.f},
      {4.f, 5.f, 0.f},
  });

  nn::ReLU relu;
  REQUIRE(is_close_mat(relu.forward(input), base, EPSILON_FLT));
}

TEST_CASE("ReLU backward", "[back_relu]")
{
  nn::matrix<float> input({
      {-1.f, -2.f, 3.f},
      {4.f, 5.f, -6.f},
  });

  nn::matrix<float> base({
      {0.f, 0.f, 3.f},
      {4.f, 5.f, 0.f},
  });
  nn::ReLU relu;

  auto result = relu.forward(input);
  REQUIRE(is_close_mat(relu.backward(input), base, EPSILON_FLT));
}

TEST_CASE("ReLU Image forward", "[relu_im]")
{
  // TODO:
}

TEST_CASE("ReLU Image backward", "[back_relu_im]")
{
  // TODO:
}

TEST_CASE("Softmax", "[softmax]")
{
  nn::matrix<float> input({
      {3.0f},
      {1.0f},
      {0.2f},
  });

  nn::matrix<float> base({
      {0.8360188f},
      {0.11314284f},
      {0.05083836f},
  });

  nn::Softmax softmax;
  REQUIRE(is_close_mat(softmax.forward(input), base, EPSILON_FLT));
}

TEST_CASE("Leaky ReLU", "[leaky_relu]")
{
  nn::matrix<float> input({
      {-1.f, -2.f, 3.f},
      {4.f, 5.f, -6.f},
  });

  float slope = 0.01f;
  nn::LeakyReLU lrelu(slope);
  nn::matrix<float> base({
      {(-1.f * slope), (-2.f * slope), 3.f},
      {4.f, 5.f, (-6.f * slope)},
  });

  REQUIRE(is_close_mat(lrelu.forward(input), base, EPSILON_FLT));
}

TEST_CASE("LeakyReLU backward", "[back_leaky_relu]")
{
  nn::matrix<float> input({
      {-1.f, -2.f, 3.f},
      {4.f, 5.f, -6.f},
  });

  float slope = 0.01f;
  nn::LeakyReLU lrelu(slope);
  nn::matrix<float> base({
      {slope, slope, 3.f},
      {4.f, 5.f, slope},
  });

  auto result = lrelu.forward(input);
  REQUIRE(is_close_mat(lrelu.backward(input), base, EPSILON_FLT));
}