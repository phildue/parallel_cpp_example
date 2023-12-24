#include "compute_photometric_error.h"
#include <Eigen/Dense>
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
int main() {

  cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::SHARED));

  const cv::Mat rgb0 = cv::imread(RESOURCE_DIR "/rgb0.png", cv::IMREAD_GRAYSCALE);
  if (rgb0.cols <= 1 || rgb0.rows <= 1) {
    throw std::runtime_error("Could not read file!");
  }

  const cv::Mat rgb1 = cv::imread(RESOURCE_DIR "/rgb1.png", cv::IMREAD_GRAYSCALE);
  if (rgb1.cols <= 1 || rgb1.rows <= 1) {
    throw std::runtime_error("Could not read file!");
  }

  Eigen::Matrix3d K;
  K << 525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1;

  const Eigen::Matrix3d Kinv = K.inverse();

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

  Eigen::Vector3d t{0, 0, 0};
  std::vector<double> ts(10);
  for (int i = 0; i < 10; i++) {

    auto start = std::chrono::high_resolution_clock::now();
    auto err = computePhotometricError(rgb0, rgb1, rgb0.cols, rgb0.rows, 1, K, R, t);

    auto end = std::chrono::high_resolution_clock::now();
    ts[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Took: [" << ts[i] << "] ms to compute for size [" << rgb0.cols << "," << rgb0.rows << "]"
              << " error: [" << err << "]" << std::endl;
  }
}
