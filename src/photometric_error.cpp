#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
namespace exec = std::execution;
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/core/cuda.hpp>
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
typedef Eigen::Matrix<double, 4, 4> Mat4f;

cv::Mat convertDepthMat(const cv::Mat& depth_, float factor);

static const std::vector<std::string> METHODS = {"parallel", "sequential", "par_two_step", "classic"};

struct Args {
  float sx = 1.0;
  float sy = 1.0;
  int N = 100;
  std::string method = "parallel";
  static Args parse(int argc, char* argv[]);
};
namespace photometric_error {
std::function<double(const cv::Mat&, const cv::Mat&)> construct(const Args& args, const cv::Mat& Z0);

}

template <typename Iterable>
double mean(const Iterable& iterable) {
  return std::reduce(iterable.begin(), iterable.end()) / std::distance(iterable.begin(), iterable.end());
}
template <typename Iterable>
double stddev(const Iterable& iterable) {
  return std::sqrt(std::transform_reduce(
           iterable.begin(), iterable.end(), 0., std::plus{}, [mean = mean(iterable)](auto x) { return std::pow(x - mean, 2); })) /
         (std::distance(iterable.begin(), iterable.end()) - 1);
}
int main(int argc, char* argv[]) {

  const Args args = Args::parse(argc, argv);

  cv::Mat I0_ = cv::imread(RESOURCE_DIR "/rgb0.png", cv::IMREAD_GRAYSCALE);
  cv::Mat Z0_ = convertDepthMat(cv::imread(RESOURCE_DIR "/depth0.png", cv::IMREAD_ANYDEPTH), 1.0 / 5000.0);
  cv::Mat I1_ = cv::imread(RESOURCE_DIR "/rgb1.png", cv::IMREAD_GRAYSCALE);

  // We have to allocate memory ourselves as otherwise its not "unified memory"
  std::vector<uint8_t> i0v(I0_.rows * I0_.cols * args.sx * args.sy);
  std::vector<uint8_t> i1v(I0_.rows * I0_.cols * args.sx * args.sy);
  std::vector<float> z0v(I0_.rows * I0_.cols * args.sx * args.sy);
  cv::Mat I0{I0_.rows * args.sy, I0_.cols * args.sx, CV_8U, i0v.data()};
  cv::Mat Z0{I0_.rows * args.sy, I0_.cols * args.sx, CV_32F, z0v.data()};
  cv::Mat I1{I0_.rows * args.sy, I0_.cols * args.sx, CV_8U, i1v.data()};

  cv::resize(I0_, I0, cv::Size{0, 0}, args.sx, args.sy);
  cv::resize(Z0_, Z0, cv::Size{0, 0}, args.sx, args.sy, cv::INTER_NEAREST);  // No bilinear interpolation for depth
  cv::resize(I1_, I1, cv::Size{0, 0}, args.sx, args.sy);

  auto computePhotometricError = photometric_error::construct(args, Z0);

  using timer = std::chrono::high_resolution_clock;

  std::cout << "Running for [" << args.N << "] iterations on scale: [" << I0.cols << "," << I0.rows << "] with method: [" << args.method
            << "]" << std::endl;
  std::vector<double> dt(args.N);
  double err = 0.;
  for (int i = 0; i < args.N; i++) {
    auto t0 = timer::now();

    err = computePhotometricError(I0, I1);

    auto t1 = timer::now();
    dt[i] = (t1 - t0).count() / 1e9;
  }
  std::cout << "Runtime = " << mean(dt) << " +- " << stddev(dt) << " Error: " << err << std::endl;
  return 0;
}

Args Args::parse(int argc, char* argv[]) {
  Args args{};
  if (argc > 1) {
    args.N = std::stoi(argv[1]);
  }
  if (argc > 2) {
    args.sx = std::stof(argv[2]);
    args.sy = std::stof(argv[2]);
  }
  if (argc > 3) {
    args.method = std::string(argv[3]);
    if (!std::any_of(METHODS.begin(), METHODS.end(), [method = args.method](auto m) { return m == method; })) {
      args.method = "parallel";
    }
  }
  return args;
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
namespace photometric_error {
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
         255.f;
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
         255.f;
}
}  // namespace par_two_step
namespace sequential {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  auto X = stdv::iota(0, I0.cols * I0.rows);
  return std::transform_reduce(
           X.begin(),
           X.end(),
           0.,
           std::plus<float>{},
           [I0 = I0.data, I1 = I1.data, w = I0.cols, h = I0.rows, reproject](int x) {
             const int u0 = x % w;
             const int v0 = (x - u0) / w;
             const Vec2f uv1 = reproject({u0, v0});
             const bool withinImage = 0 < uv1(0) && uv1(0) < w && 0 < uv1(1) && uv1(1) < h;
             return withinImage ? (float)(I1[(int)uv1(1) * w + (int)uv1(0)]) - (float)(I0[v0 * w + u0]) : 0.f;
           }) /
         255.f;
}
}  // namespace sequential

namespace classic {
template <typename Reproject>
double compute(Reproject reproject, const cv::Mat& I0, const cv::Mat& I1) {

  float r = 0.;
  for (int v0 = 0; v0 < I0.rows; v0++) {
    for (int u0 = 0; u0 < I0.cols; u0++) {
      const Vec2f uv1 = reproject({u0, v0});
      const bool withinImage = 0 < uv1(0) && uv1(0) < I0.cols && 0 < uv1(1) && uv1(1) < I0.rows;

      r += withinImage ? (float)(I1.at<uint8_t>((int)uv1(1), (int)uv1(0))) - (float)(I0.at<uint8_t>(v0, u0)) : 0.;
    }
  }
  return r / 255.f;
}
}  // namespace classic

std::function<double(const cv::Mat&, const cv::Mat&)> construct(const Args& args, const cv::Mat& Z0) {
  // TUM-RGBD Camera
  auto reproject = [fx = 525.0f * args.sx,
                    fy = 525.0f * args.sy,
                    cx = 319.5f * args.sx,
                    cy = 239.5f * args.sy,
                    w = (int)(640 * args.sx),
                    h = (int)(480 * args.sy),
                    Z0 = (float*)Z0.data,
                    pose = Mat4f::Identity()](const Vec2f& uv0) -> Vec2f {
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
    return {(fx * p0t(0) / p0t(2)) + cx, (fy * p0t(1) / p0t(2)) + cy};
  };
  const std::map<std::string, std::function<double(const cv::Mat&, const cv::Mat&)>> compute = {
    {"parallel", [reproject](auto a, auto b) { return parallel::compute(reproject, a, b); }},
    {"sequential", [reproject](auto a, auto b) { return sequential::compute(reproject, a, b); }},
    {"par_two_step", [reproject](auto a, auto b) { return par_two_step::compute(reproject, a, b); }},
    {"classic", [reproject](auto a, auto b) { return classic::compute(reproject, a, b); }},
  };
  return compute.at(args.method);
}
}  // namespace photometric_error