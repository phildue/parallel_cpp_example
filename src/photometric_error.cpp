#include <Eigen/Core>
// sudo sed -i 's/#if EIGEN_COMP_CLANG || EIGEN_COMP_CASTXML/#if EIGEN_COMP_CLANG || EIGEN_COMP_CASTXML || __NVCOMPILER_LLVM__/'
// /usr/local/include/eigen3/Eigen/src/Core/arch/NEON/Complex.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatXui8;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatXi;
typedef std::vector<MatXi> MatXiVec;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;
typedef std::vector<MatXd> MatXdVec;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXf;

template <typename Derived, int nRows, int nCols>
using Mat = Eigen::Matrix<Derived, nRows, nCols>;

template <typename Derived, int nRows>
using Vec = Eigen::Matrix<Derived, nRows, 1>;

template <int nRows, int nCols>
using Matd = Eigen::Matrix<double, nRows, nCols>;

template <int nRows, int nCols>
using Matf = Eigen::Matrix<float, nRows, nCols>;

typedef Eigen::VectorXd VecXd;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<double, 2, 2> Mat2d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef Eigen::Matrix<double, 7, 1> Vec7d;
typedef Eigen::Matrix<double, 6, 6> Mat6d;
typedef Eigen::Matrix<double, 12, 1> Vec12d;
typedef Eigen::Matrix<double, 12, 12> Mat12d;

class Camera {
public:
  Camera(double fx, double fy, double cx, double cy, int width, int height);

  std::vector<Vec2d> imageCoordinates() const;
  Vec2d project(const Vec3d& uv) const;
  Vec3d reconstruct(const Vec2d& uv, double z = 1.0) const;
  bool withinImage(const Vec2d& uv, double border = 0.) const;

  const double& fx() const { return _K(0, 0); }
  const double& fy() const { return _K(1, 1); }
  const int& width() const { return _w; }
  const int& height() const { return _h; }

  const double& cx() const { return _K(0, 2); }
  const double& cy() const { return _K(1, 2); }

  const Mat3d& K() const { return _K; }
  const Mat3d& Kinv() const { return _Kinv; }

private:
  Mat3d _K;     //< Intrinsic camera matrix
  Mat3d _Kinv;  //< Intrinsic camera matrix inverted
  int _w, _h;
};

Vec2d Camera::project(const Vec3d& p) const {
  const Vec2d invalid{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  if (p.z() <= 0) {
    return invalid;
  }

  const Vec2d uv{(fx() * p(0) / p(2)) + cx(), (fy() * p(1) / p(2)) + cy()};

  return withinImage(uv, 0.01) ? uv : invalid;
}

Vec3d Camera::reconstruct(const Vec2d& uv, double z) const { return Vec3d((uv(0) - cx()) / fx() * z, (uv(1) - cy()) / fy() * z, z); }

bool Camera::withinImage(const Vec2d& uv, double border) const {
  const int bh = std::max<int>(0, (int)border * _h);
  const int bw = std::max<int>(0, (int)border * _w);

  return (bw < uv(0) && uv(0) < _w - bw && bh < uv(1) && uv(1) < _h - bh);
}

Camera::Camera(double fx, double fy, double cx, double cy, int width, int height) : _w(width), _h(height) {
  _K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  _Kinv << 1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1;
}

std::vector<Vec2d> Camera::imageCoordinates() const {
  std::vector<Vec2d> uv(_w * _h);
  std::generate(uv.begin(), uv.end(), [u = 0, v = 0, w = _w]() mutable {
    if (u >= w) {
      u = 0;
      v++;
    }
    return Vec2d{u++, v};
  });
  return uv;
}

cv::Mat convertDepthMat(const cv::Mat& depth_, float factor) {
  cv::Mat depth(cv::Size(depth_.cols, depth_.rows), CV_32FC1);
  for (int u = 0; u < depth_.cols; u++) {
    for (int v = 0; v < depth_.rows; v++) {
      const ushort d = depth_.at<ushort>(v, u);
      depth.at<float>(v, u) = factor * static_cast<float>(d > 0 ? d : std::numeric_limits<ushort>::quiet_NaN());
    }
  }
  return depth;
}

double compute(const Camera& cam, const cv::Mat& I0, const cv::Mat& Z0, const Mat4d& pose, const cv::Mat& I1) {

  std::vector<Vec2d> uv0 = cam.imageCoordinates();
  // std::vector<unsigned char> rgb0v(rgb0.begin<unsigned char>(), rgb0.end<unsigned char>());
  std::vector<float> r(uv0.size());
  std::transform(
    std::execution::par_unseq,
    uv0.begin(),
    uv0.end(),
    r.begin(),
    [Z0 = (float*)Z0.data, I0 = (uint8_t*)I0.data, I1 = (uint8_t*)I1.data, w = cam.width(), cam, pose](auto uv0x) {
      const auto invalid = std::numeric_limits<float>::quiet_NaN();
      const int u0 = uv0x(0);
      const int v0 = uv0x(1);

      const float z = Z0[v0 * w + u0];
      if (!std::isfinite(z) || z <= 0) {
        return invalid;
      }
      const Vec3d p0 = cam.reconstruct(uv0x, z);
      /*Need to apply the transformation "manually" as otherwise cuda version does not compile due to "unsupported operation"*/
      const Vec3d p0t = {
        pose(0, 0) * p0(0) + pose(0, 1) * p0(1) + pose(0, 2) * p0(2) + pose(0, 3),
        pose(1, 0) * p0(0) + pose(1, 1) * p0(1) + pose(1, 2) * p0(2) + pose(1, 3),
        pose(2, 0) * p0(0) + pose(2, 1) * p0(1) + pose(2, 2) * p0(2) + pose(2, 3)};
      const Vec2i uv1x = Vec2d(cam.project(p0t)).cast<int>();
      if (!uv1x.allFinite()) {
        return invalid;
      }
      const float i1x = I1[uv1x(1) * w + uv1x(0)];
      const float i0x = I0[v0 * w + u0];
      const float r = (i1x - i0x);
      return r;
    });
  r.erase(std::remove_if(r.begin(), r.end(), [](auto rx) { return !std::isfinite(rx); }), r.end());
  return std::accumulate(r.begin(), r.end(), 0.) / (r.size() * 255);
}

using timer = std::chrono::high_resolution_clock;
int main(int argc, char* argv[]) {
  const cv::Mat I0 = cv::imread(RESOURCE_DIR "/rgb0.png", cv::IMREAD_GRAYSCALE);
  const cv::Mat Z0 = convertDepthMat(cv::imread(RESOURCE_DIR "/depth0.png", cv::IMREAD_ANYDEPTH), 1.0 / 5000.0);
  const cv::Mat I1 = cv::imread(RESOURCE_DIR "/rgb1.png", cv::IMREAD_GRAYSCALE);
  Camera cam{525.0, 525.0, 319.5, 239.5, 640, 480};
  Mat4d motion = Mat4d::Identity();
  int N = 100;
  VecXd dt = VecXd::Zero(N);
  for (int i = 0; i < N; i++) {
    auto t0 = timer::now();
    auto error = compute(cam, I0, Z0, motion, I1);
    auto t1 = timer::now();
    dt(i) = (t1 - t0).count() / 1e9;
    std::cout << "Execution on took " << dt(i) << "s error: [" << error << "]" << std::endl;
  }
  std::cout << "Mean = " << dt.mean() << std::endl;
  return 0;
}
