#include <opencv2/core.hpp>

#include "layers/lrn.h"

using namespace nn;

LocalResponseNorm::LocalResponseNorm(float alpha,
                                     float beta,
                                     float k,
                                     size_t local_size)
    : m_alpha(alpha)
    , m_beta(beta)
    , m_k(k)
    , m_size(local_size)
{
}

cv::Mat LocalResponseNorm::forward(const cv::Mat& input)
{
  // TODO:
}

cv::Mat LocalResponseNorm::backward(const cv::Mat& grad)
{
  // TODO:
}