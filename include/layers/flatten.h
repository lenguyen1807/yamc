#ifndef FLATTEN_H
#define FLATTEN_H

#include <opencv2/core.hpp>

#include "layer.h"
#include "matrix.h"

namespace nn
{
class Flatten : public Layer<float>
{
public:
  Flatten() = default;

  // Flatten will "flatten" an image to 1D vector (matrix class)
  matrix<float> forward_im(const cv::Mat& im);

  // The backward should return an image from a vector
  // I'm so stupid so I must create another backward function with different
  // return type :>
  cv::Mat backward_im(const matrix<float>& mat);

private:
  size_t m_channels;
  size_t m_height;
  size_t m_width;
};
}  // namespace nn

#endif  // FLATTEN_H