#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
namespace exec = std::execution;
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
namespace stdv = std::views;
namespace stdr = std::ranges;

#include <stdexcept>
#include <vector>
typedef Eigen::VectorXf VecXf;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Matrix<double, 2, 2> Mat2f;
typedef Eigen::Matrix<double, 3, 3> Mat3f;
typedef Eigen::Matrix<double, 4, 4> Mat4f;

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

namespace parallel {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  auto idxx = stdv::iota(0, I0.cols * I0.rows);
  return std::transform_reduce(
           exec::par_unseq,
           idxx.begin(),
           idxx.end(),
           0.,
           std::plus<float>{},
           [I0 = I0.data, I1 = I1.data, w = I0.cols, h = I0.rows, reproject](int idx) {
             const int u0 = idx % w;
             const int v0 = (idx - (idx % w)) / w;
             const Vec2f uv1 = reproject({u0, v0});
             const bool withinImage = 0 < uv1(0) && uv1(0) < w && 0 < uv1(1) && uv1(1) < h;
             return withinImage ? (float)(I1[(int)uv1(1) * w + (int)uv1(0)]) - (float)(I0[v0 * w + u0]) : 0.f;
           }) /
         255.;
}
}  // namespace parallel
namespace par_two_step {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  auto idxx = stdv::iota(0, I0.cols * I0.rows);
  std::vector<Vec2i> uv1(I0.cols * I0.rows);
  std::transform(exec::par_unseq, idxx.begin(), idxx.end(), uv1.begin(), [reproject, w = I0.cols](int idx) {
    return Vec2f(reproject({idx % w, (idx - (idx % w)) / w})).cast<int>();
  });
  return std::transform_reduce(
           exec::par_unseq,
           idxx.begin(),
           idxx.end(),
           0.,
           std::plus<float>{},
           [I0 = I0.data, I1 = I1.data, w = I0.cols, h = I0.rows, uv1 = uv1.data()](int idx) {
             const int u0 = idx % w;
             const int v0 = (idx - u0) / w;
             const Vec2i uv1x = uv1[idx];
             const bool withinImage = 0 < uv1x(0) && uv1x(0) < w && 0 < uv1x(1) && uv1x(1) < h;
             return withinImage ? (float)(I1[uv1x(1) * w + uv1x(0)]) - (float)(I0[v0 * w + u0]) : 0.f;
           }) /
         255.;
}
}  // namespace par_two_step
namespace sequential {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  auto idxx = stdv::iota(0, I0.cols * I0.rows);
  return std::transform_reduce(
           idxx.begin(),
           idxx.end(),
           0.,
           std::plus<float>{},
           [I0 = I0.data, I1 = I1.data, w = I0.cols, h = I0.rows, reproject](int idx) {
             const int u0 = idx % w;
             const int v0 = (idx - u0) / w;
             const Vec2f uv1 = reproject({u0, v0});
             const bool withinImage = 0 < uv1(0) && uv1(0) < w && 0 < uv1(1) && uv1(1) < h;
             return withinImage ? (float)(I1[(int)uv1(1) * w + (int)uv1(0)]) - (float)(I0[v0 * w + u0]) : 0.f;
           }) /
         255.;
}
}  // namespace sequential

namespace classic {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  double r = 0.;
  for (int u0 = 0; u0 < I0.cols; u0++) {
    for (int v0 = 0; v0 < I0.rows; v0++) {
      const Vec2f uv1 = reproject({u0, v0});
      const bool withinImage = 0 < uv1(0) && uv1(0) < I0.cols && 0 < uv1(1) && uv1(1) < I0.rows;

      r += withinImage ? (double)(I1.at<uint8_t>((int)uv1(1), (int)uv1(0))) - (double)(I0.at<uint8_t>(v0, u0)) : 0.;
    }
  }
  return r / 255.;
}
}  // namespace classic
int main(int argc, char* argv[]) {
  float sx = 1.0;
  float sy = 1.0;
  int N = 100;
  std::string method = "classic";
  if (argc > 1) {
    N = std::stoi(argv[1]);
  }
  if (argc > 2) {
    sx = std::stof(argv[2]);
    sy = std::stof(argv[2]);
  }
  if (argc > 3) {
    method = std::string(argv[3]);
  }

  cv::Mat I0 = cv::imread(RESOURCE_DIR "/rgb0.png", cv::IMREAD_GRAYSCALE);
  cv::Mat Z0 = convertDepthMat(cv::imread(RESOURCE_DIR "/depth0.png", cv::IMREAD_ANYDEPTH), 1.0 / 5000.0);
  cv::Mat I1 = cv::imread(RESOURCE_DIR "/rgb1.png", cv::IMREAD_GRAYSCALE);

  cv::resize(I0, I0, cv::Size{0, 0}, sx, sy);
  cv::resize(Z0, Z0, cv::Size{0, 0}, sx, sy, cv::INTER_NEAREST);
  cv::resize(I1, I1, cv::Size{0, 0}, sx, sy);

  /*FIXME: the memory allocated by opencv is not using unified memory*/
  std::vector<uint8_t> I0d(I0.begin<uint8_t>(), I0.end<uint8_t>());
  std::vector<uint8_t> I1d(I1.begin<uint8_t>(), I1.end<uint8_t>());
  std::vector<float> Z0d(Z0.begin<float>(), Z0.end<float>());
  cv::Mat I0_{I0.rows, I0.cols, CV_8U, I0d.data()};
  cv::Mat Z0_{I0.rows, I0.cols, CV_32F, Z0d.data()};
  cv::Mat I1_{I0.rows, I0.cols, CV_8U, I1d.data()};
  Mat4f pose = Mat4f::Identity();

  // TUM-RGBD Dataset
  auto reproject =
    [fx = 525.0f, fy = 525.0f, cx = 319.5f, cy = 239.5f, w = 640, h = 480, Z0 = Z0d.data(), pose](const Vec2f& uv0) -> Vec2f {
    const float z = Z0[(int)uv0(1) * w + (int)uv0(0)];
    if (!std::isfinite(z) || z <= 0) {
      return {-1, -1};
    }
    const Vec3f p0{(uv0(0) - cx) / fx * z, (uv0(1) - cy) / fy * z, z};

    /*Need to apply the transformation "manually" as otherwise cuda version does not compile due to "unsupported operation"*/
    const Vec3f p0t = {
      pose(0, 0) * p0(0) + pose(0, 1) * p0(1) + pose(0, 2) * p0(2) + pose(0, 3),
      pose(1, 0) * p0(0) + pose(1, 1) * p0(1) + pose(1, 2) * p0(2) + pose(1, 3),
      pose(2, 0) * p0(0) + pose(2, 1) * p0(1) + pose(2, 2) * p0(2) + pose(2, 3)};

    if (p0t(2) <= 0) {
      return {-1, -1};
    }
    const Vec2f uv1{(fx * p0t(0) / p0t(2)) + cx, (fy * p0t(1) / p0t(2)) + cy};
    return uv1;
  };
  std::function<double(const cv::Mat&, const cv::Mat&)> compute;
  if (method == "parallel") {
    compute = [reproject](auto a, auto b) { return parallel::compute(reproject, a, b); };
  } else if (method == "sequential") {
    compute = [reproject](auto a, auto b) { return sequential::compute(reproject, a, b); };
  } else if (method == "par_two_step") {
    compute = [reproject](auto a, auto b) { return par_two_step::compute(reproject, a, b); };
  } else {
    method = "classic";
    compute = [reproject](auto a, auto b) { return classic::compute(reproject, a, b); };
  }
  std::cout << "Running for [" << N << "] iterations on scale: [" << I0.cols << "," << I0.rows << "] with method: [" << method << "]"
            << std::endl;
  std::vector<double> dt(N);
  double err = 0.;
  for (int i = 0; i < N; i++) {
    auto t0 = timer::now();

    err = compute(I0_, I1_);

    auto t1 = timer::now();
    dt[i] = (t1 - t0).count() / 1e9;
  }
  std::cout << "Mean = " << std::reduce(dt.begin(), dt.end()) / N << " Error: " << err << std::endl;
  return 0;
}
