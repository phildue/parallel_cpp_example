#include "compute_photometric_error.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
double computePhotometricError(
  const cv::Mat& rgb0,
  const cv::Mat& rgb1,
  int cols,
  int rows,
  int windowSize,
  const Eigen::Matrix3d& K,
  const Eigen::Matrix3d& R,
  const Eigen::Vector3d& t) {

  struct Idx2d {
    int u, v;
  };
  std::vector<Idx2d> uv(rows * cols);
  std::generate(uv.begin(), uv.end(), [u = 0, v = 0, cols]() mutable {
    if (u >= cols) {
      u = 0;
      v++;
    }
    return Idx2d{u++, v};
  });

//  std::vector<unsigned char> rgb0v(rgb0.begin<unsigned char>(), rgb0.end<unsigned char>());
//  std::vector<unsigned char> rgb1v(rgb1.begin<unsigned char>(), rgb1.end<unsigned char>());
  std::vector<double> err(rows * cols);
  std::transform(
    std::execution::par_unseq,
    uv.begin(),
    uv.end(),
    err.begin(),
    [=, tx = t(0), ty = t(1), tz = t(2), Kinv = K.inverse(), rgb0 = rgb0.data, rgb1 = rgb1.data](auto uv) {
      double residual = 0;
      for (int i = uv.v - std::floor(windowSize / 2); i <= uv.v + std::ceil(windowSize / 2); i++) {
        for (int j = uv.u - std::floor(windowSize / 2); j <= uv.u + std::ceil(windowSize / 2); j++) {
          const int v0 = std::max<int>(0, std::min<int>(i, rows - 1));
          const int u0 = std::max<int>(0, std::min<int>(j, cols - 1));
          const Eigen::Vector3i uv0{u0, v0, 1};
          const double intensity0 = rgb0[uv0.y() * cols + uv0.x()];
          const Eigen::Vector3d uv1 = Eigen::Vector3d{tx, ty, tz};
          const int v1 = std::max<int>(0, std::min<int>(uv1(0) / uv1(2), rows - 1));
          const int u1 = std::max<int>(0, std::min<int>(uv1(1) / uv1(2), cols - 1));
          // const Eigen::Vector3d uv1 = K * Eigen::Matrix3d{Eigen::Matrix3d{R * (Kinv * uv0)}.colwise() + t};
          const double intensity1 = rgb1[v1 * cols + u1];
          residual += std::abs(intensity0 - intensity1);
        }
      }
      return std::isfinite(residual) ? residual : 0.;
    });
  return std::accumulate(err.begin(), err.end(), 0.) / (rows * cols * 255);
}
