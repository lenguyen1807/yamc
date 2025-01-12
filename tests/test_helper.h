#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <cmath>

#include <opencv2/core.hpp>

#include "matrix.h"

#define EPSILON_STD(T) std::numeric_limits<T>::epsilon()
#define EPSILON_FLT 0.001f

// https://stackoverflow.com/questions/4548004/how-to-correctly-and-standardly-compare-floats
template<typename T>
static bool are_equal(T f1, T f2, T epsilon)
{
  return (std::fabs(f1 - f2)
          <= epsilon * std::fmax(std::fabs(f1), std::fabs(f2)));
}

bool is_close_mat(const nn::matrix<float>& mat1,
                  const nn::matrix<float>& mat2,
                  float epsilon);

// https://stackoverflow.com/questions/67890246/how-to-check-if-two-images-are-almost-the-same-in-opencv
bool is_close_im(const cv::Mat& im1,
                 const cv::Mat& im2,
                 float xy_threshold = .1f);

#endif  // TEST_HELPER_H