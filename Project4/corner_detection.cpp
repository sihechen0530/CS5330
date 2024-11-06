/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 4
 * main function
 */

#include "calibration.h"
#include "utils.h"

constexpr char kQuitKey = 'q';
constexpr char kSaveKey = 's';

constexpr char kOriginal[] = "original";
constexpr char kResult[] = "result";
constexpr char kSaveFilePath[] = "corners.txt";

constexpr int kMinCalibPointCount = 5;
constexpr int kCornerRow = 6;
constexpr int kCornerCol = 9;

// main function
int main(int argc, char *argv[]) {
  cv::VideoCapture camera(0);
  // open the video device
  if (!camera.isOpened()) {
    LOG_INFO("Unable to open video device");
    return (-1);
  }

  camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

  // get some properties of the image
  cv::Size image_size((int)camera.get(cv::CAP_PROP_FRAME_WIDTH),
                      (int)camera.get(cv::CAP_PROP_FRAME_HEIGHT));
  LOG_INFO("Expected size: " << image_size.width << " " << image_size.height);

  cv::Mat frame;
  std::vector<std::vector<cv::Vec3f>> point_list;
  std::vector<std::vector<cv::Point2f>> corner_list;
  cv::Mat cameraMatrix;
  cv::Mat rvec, tvec;
  bool is_intrinsic_params_ready = false;
  double rms = 0.0f;
  for (;;) {
    camera >> frame;
    if (frame.empty()) {
      LOG_ERROR("Error: No frame captured.");
      break;
    }
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Point2f> corner_set;
    if (0 != getChessBoardCorners(frame, kCornerRow, kCornerCol, &result,
                                  &point_set, &corner_set)) {
      LOG_ERROR("failed to get chessboard corners");
      continue;
    }
    if (corner_list.size() >= kMinCalibPointCount) {
      LOG_INFO("calib point count: "
               << corner_list.size() << " reach required calib point count "
               << kMinCalibPointCount << " do calibrating...");
      calibrate(point_list, corner_list, image_size, &cameraMatrix);
      is_intrinsic_params_ready = true;
      point_list.clear();
      corner_list.clear();
    }
    if (is_intrinsic_params_ready) {
      getPose(point_set, corner_set, cameraMatrix, &rvec, &tvec);
      projectPoints(rvec, tvec, cameraMatrix, &frame);
    }
    cv::imshow(kOriginal, frame);
    cv::imshow(kResult, result);
    char new_key = cv::waitKey(10);
    if (new_key >= 0) {
      if (kQuitKey == new_key) {
        break;
      }
      if (kSaveKey == new_key) {
        if (point_set.size() == corner_set.size() &&
            corner_set.size() == kCornerRow * kCornerCol) {
          LOG_INFO("save #" << corner_set.size() << " corner");
          point_list.push_back(point_set);
          corner_list.push_back(corner_set);
        }
      }
    }
  }

  camera.release();
  cv::destroyAllWindows();
  saveCornersToFile(point_list, corner_list, kSaveFilePath);
  // loadCornersFromFile(kSaveFilePath, &point_list, &corner_list);
  return 0;
}
