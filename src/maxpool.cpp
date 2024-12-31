#include "maxpool.h"

using namespace nn;

Maxpool::Maxpool(size_t size, size_t stride)
    : m_size(size)
    , m_stride(stride)
{
}

matrix<float> Maxpool::forward(const matrix<float>& input)
{
  m_input = input;
}