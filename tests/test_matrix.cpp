#include <catch2/catch_test_macros.hpp>

#include "matrix.h"
#include "test_helper.h"

TEST_CASE("Matrix multiplication", "[matmul]")
{
  // array([[1, 2],
  //  [3, 4],
  //  [5, 6]])
  nn::matrix<float> mat1({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});

  // array([[1, 2, 3],
  //  [4, 5, 6]])
  nn::matrix<float> mat2({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

  // array([[ 9, 12, 15],
  //  [19, 26, 33],
  //  [29, 40, 51]])
  nn::matrix<float> base(
      {{9.0f, 12.0f, 15.0f}, {19.0f, 26.0f, 33.0f}, {29.0f, 40.0f, 51.0f}});

  nn::matrix<float> mat3 = mat1 * mat2;
  REQUIRE(is_close_mat(mat3, base, EPSILON_FLT));
}

TEST_CASE("Scalar multiplication", "[scalarmul]")
{
  // array([[1, 2],
  //  [3, 4],
  //  [5, 6]])
  nn::matrix<float> mat1({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  nn::matrix<float> base({{2.f, 4.f}, {6.f, 8.f}, {10.f, 12.f}});
  REQUIRE(is_close_mat(mat1 % 2.f, base, EPSILON_FLT));
}

// TODO: Add more tests later