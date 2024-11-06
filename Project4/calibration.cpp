#include "calibration.h"
#include <fstream>

int getChessBoardCorners(const cv::Mat &src, const int corner_row,
                         const int corner_col, cv::Mat *dst,
                         std::vector<cv::Vec3f> *point_set,
                         std::vector<cv::Point2f> *corner_set) {
  cv::Size patternsize(corner_col, corner_row); // interior number of corners
  bool patternfound = cv::findChessboardCorners(
      *dst, patternsize, *corner_set,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
          cv::CALIB_CB_FAST_CHECK);
  if (!patternfound) {
    LOG_ERROR("pattern not found");
    return -1;
  }
  cv::cornerSubPix(
      *dst, *corner_set, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                       0.1));
  for (int y = 0; y < corner_row; ++y) {
    for (int x = 0; x < corner_col; ++x) {
      point_set->push_back({x, -y, 0});
    }
  }
  cv::drawChessboardCorners(
      *dst, patternsize, cv::Mat(*corner_set),
      patternfound); // see if there is a waiting keystroke
  return 0;
}

int saveCornersToFile(const std::vector<std::vector<cv::Vec3f>> &point_list,
                      const std::vector<std::vector<cv::Point2f>> &corner_list,
                      const char *file_path) {
  ASSERT_EQ_RET(point_list.size(), corner_list.size(), -1);
  std::ofstream outFile(file_path);
  ASSERT_RET(outFile, -1);
  outFile << std::fixed << std::setprecision(6);

  int n = point_list.size();
  outFile << n << "\n";
  for (size_t i = 0; i < n; ++i) {
    outFile << point_list[i].size() << "\n";
    for (size_t j = 0; j < point_list[i].size(); ++j) {
      outFile << point_list[i][j][0] << " " << point_list[i][j][1] << " "
              << point_list[i][j][2] << "\n";
    }
    outFile << corner_list[i].size() << "\n";
    for (size_t j = 0; j < corner_list[i].size(); ++j) {
      outFile << corner_list[i][j].x << " " << corner_list[i][j].y << "\n";
    }
  }

  outFile.close();
  return 0;
}

int loadCornersFromFile(const char *file_path,
                        std::vector<std::vector<cv::Vec3f>> *point_list,
                        std::vector<std::vector<cv::Point2f>> *corner_list) {
  std::ifstream inFile(file_path);
  ASSERT_RET(inFile, -1);

  int n;
  inFile >> n;
  point_list->resize(n);
  corner_list->resize(n);

  for (int i = 0; i < n; ++i) {
    int point_size;
    inFile >> point_size;
    (*point_list)[i].resize(point_size);
    for (int j = 0; j < point_size; ++j) {
      inFile >> (*point_list)[i][j][0] >> (*point_list)[i][j][1] >>
          (*point_list)[i][j][2];
    }

    int corner_size;
    inFile >> corner_size;
    (*corner_list)[i].resize(corner_size);
    for (int j = 0; j < corner_size; ++j) {
      inFile >> (*corner_list)[i][j].x >> (*corner_list)[i][j].y;
    }
  }

  inFile.close();
  return 0;
}

int calibrate(const std::vector<std::vector<cv::Vec3f>> &point_list,
              const std::vector<std::vector<cv::Point2f>> &corner_list,
              const cv::Size image_size, cv::Mat *cameraMatrix) {
  // task 3: Print out the camera matrix and distortion coefficients before and
  // after the calibration, along with the final re-projection error. The two
  // focal lengths should be the same value, and the u0, v0 values should be
  // close to the initial estimates of the center of the image.
  *cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, image_size.width / 2.0, 0, 1,
                   image_size.height / 2.0, 0, 0, 1);
  // cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
  std::vector<double> distCoeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  LOG_INFO("Camera Matrix before calib: \n" << *cameraMatrix);
  double rms =
      cv::calibrateCamera(point_list, corner_list, image_size, *cameraMatrix,
                          distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

  // Output results
  LOG_INFO("Camera Matrix after calib: \n" << *cameraMatrix);
  // LOG_INFO("Distortion Coefficients: \n" << *distCoeffs);
  LOG_INFO(
      "RMS error: " << rms << " ("
                    << image_size.width / 2.0 - cameraMatrix->at<double>(0, 2)
                    << ","
                    << image_size.height / 2.0 - cameraMatrix->at<double>(1, 2)
                    << ")");

  // Optional: Display rotation and translation vectors for each image
  // for (size_t i = 0; i < rvecs.size(); ++i) {
  //   LOG_INFO("Rotation Vector for image " << i << ":\n" << rvecs[i]);
  //   LOG_INFO("Translation Vector for image " << i << ":\n" << rvecs[i]);
  // }
  return 0;
}

int getPose(const std::vector<cv::Vec3f> &point_set,
            const std::vector<cv::Point2f> &corner_set,
            const cv::Mat &cameraMatrix, cv::Mat *rvec, cv::Mat *tvec) {
  std::vector<double> distCoeffs;
  cv::solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, *rvec, *tvec);
  LOG_INFO("rotation: " << *rvec << " transition: " << *tvec);
  return 0;
}

int projectPoints(const cv::Mat &rvec, const cv::Mat &tvec,
                  const cv::Mat &cameraMatrix, cv::Mat *image) {
  std::vector<cv::Point3f> axis_points_3d;
  axis_points_3d.push_back(cv::Point3f(0, 0, 0)); // Origin
  axis_points_3d.push_back(cv::Point3f(1, 0, 0)); // X-axis (red)
  axis_points_3d.push_back(cv::Point3f(0, 1, 0)); // Y-axis (green)
  axis_points_3d.push_back(cv::Point3f(0, 0, 1)); // Z-axis (blue)

  // Project 3D points to 2D
  std::vector<cv::Point2f> axis_points_2d;
  cv::projectPoints(axis_points_3d, rvec, tvec, cameraMatrix, cv::Mat(),
                    axis_points_2d);

  // Draw the axes
  cv::line(*image, axis_points_2d[0], axis_points_2d[1], cv::Scalar(0, 0, 255),
           2); // X-axis in red
  cv::line(*image, axis_points_2d[0], axis_points_2d[2], cv::Scalar(0, 255, 0),
           2); // Y-axis in green
  cv::line(*image, axis_points_2d[0], axis_points_2d[3], cv::Scalar(255, 0, 0),
           2); // Z-axis in blue

  std::vector<cv::Vec3f> object_points_3d;
  object_points_3d.push_back(cv::Vec3f(2, -2, 1));
  object_points_3d.push_back(cv::Vec3f(6, -1, 3));
  object_points_3d.push_back(cv::Vec3f(6, -5, 4));
  object_points_3d.push_back(cv::Vec3f(4, -3, 5));
  std::vector<cv::Point2f> object_points_2d;
  cv::projectPoints(object_points_3d, rvec, tvec, cameraMatrix, cv::Mat(),
                    object_points_2d);
  for (const auto &point : object_points_2d) {
    cv::circle(*image, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
  }

  for (size_t i = 0; i < object_points_2d.size(); ++i) {
    for (size_t j = i + 1; j < object_points_2d.size(); ++j) {
      cv::line(*image, object_points_2d[i], object_points_2d[j],
               cv::Scalar(255, 0, 0), 2);
    }
  }

  return 0;
}
