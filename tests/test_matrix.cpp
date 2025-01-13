#include <catch2/catch_test_macros.hpp>

#include "matrix.h"
#include "test_helper.h"

TEST_CASE("Matrix multiplication", "[matmul]")
{
  // array([[1, 2],
  //  [3, 4],
  //  [5, 6]])
  nn::matrix<float> mat1({
      {1.0f, 2.0f},
      {3.0f, 4.0f},
      {5.0f, 6.0f},
  });

  // array([[1, 2, 3],
  //  [4, 5, 6]])
  nn::matrix<float> mat2({
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  });

  // array([[ 9, 12, 15],
  //  [19, 26, 33],
  //  [29, 40, 51]])
  nn::matrix<float> base({
      {9.0f, 12.0f, 15.0f},
      {19.0f, 26.0f, 33.0f},
      {29.0f, 40.0f, 51.0f},
  });

  nn::matrix<float> mat3 = mat1 * mat2;
  REQUIRE(is_close_mat(mat3, base, EPSILON_STD(float)));
}

TEST_CASE("Scalar multiplication", "[scalarmul]")
{
  // array([[1, 2],
  //  [3, 4],
  //  [5, 6]])
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {3.f, 4.f},
      {5.f, 6.f},
  });
  nn::matrix<float> base({
      {2.f, 4.f},
      {6.f, 8.f},
      {10.f, 12.f},
  });
  REQUIRE(is_close_mat(mat1 % 2.f, base, EPSILON_STD(float)));
}

TEST_CASE("Vstack", "[vstack]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> mat2({
      {2.f, 4.f},
      {6.f, 8.f},
  });
  nn::matrix<float> base({
      {1.f, 2.f},
      {2.f, 4.f},
      {2.f, 4.f},
      {6.f, 8.f},
  });
  nn::matrix<float> base2({
      {1.f, 2.f},
      {2.f, 4.f},
      {2.f, 4.f},
      {6.f, 8.f},
      {2.f, 4.f},
      {6.f, 8.f},
  });

  REQUIRE(
      is_close_mat(nn::matrix<float>::vstack(mat1, mat2), base, EPSILON_FLT));
  REQUIRE(
      is_close_mat(nn::matrix<float>::vstack(base, mat2), base2, EPSILON_FLT));
}

TEST_CASE("Hstack", "[hstack]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> mat2({
      {3.f, 5.f},
      {6.f, 9.f},
  });
  nn::matrix<float> base({
      {1.f, 2.f, 3.f, 5.f},
      {2.f, 4.f, 6.f, 9.f},
  });

  REQUIRE(
      is_close_mat(nn::matrix<float>::hstack(mat1, mat2), base, EPSILON_FLT));
}

TEST_CASE("Sum axis=0", "[sum-axis=0]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({{3.f, 6.f}});
  REQUIRE(is_close_mat(mat1.sum(0), base, EPSILON_FLT));
}

TEST_CASE("Sum axis=1", "[sum-axis=1]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({{3.f}, {6.f}});
  REQUIRE(is_close_mat(mat1.sum(1), base, EPSILON_FLT));
}

TEST_CASE("Mean axis=0", "[mean-axis=0]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({{3.f / 2.f, 6.f / 2.f}});
  REQUIRE(is_close_mat(mat1.mean(0), base, EPSILON_FLT));
}

TEST_CASE("Mean axis=1", "[mean-axis=1]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({{3.f / static_cast<float>(mat1.cols)},
                          {6.f / static_cast<float>(mat1.cols)}});
  REQUIRE(is_close_mat(mat1.mean(1), base, EPSILON_FLT));
}

TEST_CASE("Max axis=0", "[max-axis=0]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({{2.f, 4.f}});
  REQUIRE(is_close_mat(mat1.max(0), base, EPSILON_FLT));
}

TEST_CASE("Max axis=1", "[max-axis=1]")
{
  nn::matrix<float> mat1({
      {1.f, 2.f},
      {2.f, 4.f},
  });
  nn::matrix<float> base({
      {2.f},
      {4.f},
  });
  REQUIRE(is_close_mat(mat1.max(1), base, EPSILON_FLT));
}

// TODO: Add more tests later