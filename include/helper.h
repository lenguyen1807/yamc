#ifndef HELPER_H
#define HELPER_H

#include "matrix.h"

/* Initialize */
static nn::matrix<float> xavier_initialize(size_t rows, size_t cols)
{
  // Glorot/Xavier initialization
  float scale = std::sqrt(2.0f / (rows + cols));

  // Prevent scale from becoming too small or too large
  const float MIN_SCALE = 1e-4f;
  const float MAX_SCALE = 1e-1f;
  scale = std::min(std::max(scale, MIN_SCALE), MAX_SCALE);

  return nn::matrix<float>::nrand(rows, cols, 0.0f, scale);
}

static nn::matrix<float> he_initialize(size_t rows, size_t cols)
{
  float scale = std::sqrt(2.0f / cols);

  const float MIN_SCALE = 1e-4f;
  const float MAX_SCALE = 1e-1f;
  scale = std::min(std::max(scale, MIN_SCALE), MAX_SCALE);

  return nn::matrix<float>::nrand(rows, cols, 0.0f, scale);
}

/* Image normalize */
static nn::matrix<float> normalize_minmax(const nn::matrix<float>& image)
{
  float min_val = nn::matrix<float>::min(image);
  float max_val = nn::matrix<float>::max(image);
  float range = max_val - min_val;

  if (range < 1e-10f) {
    return nn::matrix<float>::values_like(0.0f, image);
  }

  return (image - min_val) / range;
}

static void print_stats(const nn::matrix<float>& m, const std::string& name)
{
  float min_val = nn::matrix<float>::min(m);
  float max_val = nn::matrix<float>::max(m);
  float mean_val = m.reduce_sum() / (m.rows * m.cols);
  std::cout << name << " stats - min: " << min_val << " max: " << max_val
            << " mean: " << mean_val << std::endl;
}

#endif  // HELPER_H
