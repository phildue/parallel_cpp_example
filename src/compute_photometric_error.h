#pragma once
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

double computePhotometricError(
  const cv::Mat& rgb0,
  const cv::Mat& rgb1,
  int cols,
  int rows,
  int windowSize,
  const Eigen::Matrix3d& K,
  const Eigen::Matrix3d& R,
  const Eigen::Vector3d& t);
