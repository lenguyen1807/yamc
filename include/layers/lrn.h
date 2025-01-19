#ifndef LRN_H
#define LRN_H

#include <opencv2/core/mat.hpp>

#include "layer.h"

namespace nn
{
class LocalResponseNorm : public Layer<float>
{
public:
  LocalResponseNorm(float alpha = 1e-4f,
                    float beta = 0.75f,
                    float k = 2.0f,
                    size_t local_size = 5);
  IMPLEMENT_LAYER_IM();

private:
  float m_alpha;
  float m_beta;
  float m_k;
  size_t m_size;

  cv::Mat m_im;
  std::vector<cv::Mat> m_square_sum;
  std::vector<cv::Mat> m_norm_factor;
};
}  // namespace nn

#endif  // LRN_H