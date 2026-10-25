#pragma once
#include <math.h>
constexpr int kWindowSize = 50;
constexpr int kAxes = 6;
constexpr int kFeatureCount = 24;

// Same axis-major order and population standard deviation as training.
inline void extractFeatures(const float window[kWindowSize][kAxes], float output[kFeatureCount]) {
  for (int axis = 0; axis < kAxes; ++axis) {
    float sum = 0, low = window[0][axis], high = low;
    for (int i = 0; i < kWindowSize; ++i) {
      const float v = window[i][axis];
      sum += v;
      if (v < low) low = v;
      if (v > high) high = v;
    }
    const float mean = sum / kWindowSize;
    float squares = 0;
    for (int i = 0; i < kWindowSize; ++i) {
      const float d = window[i][axis] - mean;
      squares += d * d;
    }
    output[axis * 4] = mean;
    output[axis * 4 + 1] = sqrtf(squares / kWindowSize);
    output[axis * 4 + 2] = low;
    output[axis * 4 + 3] = high;
  }
}