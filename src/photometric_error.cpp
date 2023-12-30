#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <stdexcept>
#include <vector>
typedef Eigen::VectorXd VecXd;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<double, 2, 2> Mat2d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;

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
using timer = std::chrono::high_resolution_clock;

double compute(const Camera& cam, const cv::Mat& I0, const cv::Mat& Z0, const cv::Mat& I1, const Mat4d& pose) {
  struct Residual {
    double r;
    int n;
  };

  namespace stdv = std::views;
  auto idxx = stdv::iota(0, I0.cols * I0.rows);
  const Residual r = std::transform_reduce(
    std::execution::par_unseq,
    idxx.begin(),
    idxx.end(),
    Residual{0., 0},
    [](auto a, auto b) {
      return Residual{a.r + b.r, a.n + b.n};
    },
    [I0 = I0.data, I1 = I1.data, Z0 = (float*)Z0.data, cam = cam, pose, w = cam.width()](int idx) {
      int u = idx % w;
      int v = (idx - u) / w;
      const float z = Z0[idx];
      if (!std::isfinite(z) || z <= 0) {
        return Residual{0, 0};
      }
      const Vec3d p0 = cam.reconstruct({u, v}, z);
      /*Need to apply the transformation "manually" as otherwise cuda version does not compile due to "unsupported operation"*/
      const Vec3d p0t = {
        pose(0, 0) * p0(0) + pose(0, 1) * p0(1) + pose(0, 2) * p0(2) + pose(0, 3),
        pose(1, 0) * p0(0) + pose(1, 1) * p0(1) + pose(1, 2) * p0(2) + pose(1, 3),
        pose(2, 0) * p0(0) + pose(2, 1) * p0(1) + pose(2, 2) * p0(2) + pose(2, 3)};
      const Vec2i uv1 = Vec2d(cam.project(p0t)).cast<int>();
      if (!uv1.allFinite()) {
        return Residual{0, 0};
      }
      const float i1x = I1[uv1(1) * cam.width() + uv1(0)];
      const float i0x = I0[idx];
      const float r = (i1x - i0x);
      return Residual{r, 1};
    });
  return r.r / (255 * r.n);
}
int main(int argc, char* argv[]) {
  float sx = 1.0;
  float sy = 1.0;
  int N = 100;

  if (argc > 1) {
    N = std::stoi(argv[1]);
  }
  if (argc > 2) {
    sx = std::stof(argv[2]);
    sy = std::stof(argv[2]);
  }

  cv::Mat I0 = cv::imread(RESOURCE_DIR "/rgb0.png", cv::IMREAD_GRAYSCALE);
  cv::Mat Z0 = convertDepthMat(cv::imread(RESOURCE_DIR "/depth0.png", cv::IMREAD_ANYDEPTH), 1.0 / 5000.0);
  cv::Mat I1 = cv::imread(RESOURCE_DIR "/rgb1.png", cv::IMREAD_GRAYSCALE);

  cv::resize(I0, I0, cv::Size{0, 0}, sx, sy);
  cv::resize(Z0, Z0, cv::Size{0, 0}, sx, sy);
  cv::resize(I1, I1, cv::Size{0, 0}, sx, sy);

  Camera cam{525.0 * sx, 525.0 * sy, 319.5 * sx, 239.5 * sy, 640 * sx, 480 * sy};
  std::vector<uint8_t> I0d(I0.begin<uint8_t>(), I0.end<uint8_t>());
  std::vector<uint8_t> I1d(I1.begin<uint8_t>(), I1.end<uint8_t>());
  std::vector<float> Z0d(Z0.begin<float>(), Z0.end<float>());
  cv::Mat I0_{I0.rows, I0.cols, CV_8U, I0d.data()};
  cv::Mat Z0_{I0.rows, I0.cols, CV_32F, Z0d.data()};
  cv::Mat I1_{I0.rows, I0.cols, CV_8U, I1d.data()};
  Mat4d pose = Mat4d::Identity();
  VecXd dt = VecXd::Zero(N);
  std::cout << "Running for [" << N << "] iterations on scale: [" << I0.cols << "," << I0.rows << "]" << std::endl;
  double err = 0.;
  for (int i = 0; i < N; i++) {
    auto t0 = timer::now();

    err = compute(cam, I0_, Z0_, I1_, pose);

    auto t1 = timer::now();
    dt(i) = (t1 - t0).count() / 1e9;
  }
  std::cout << "Mean = " << dt.mean() << " Error: " << err << std::endl;
  return 0;
}
