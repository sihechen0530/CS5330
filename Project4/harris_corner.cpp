#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>

const int kBlockSize = 2;
const int kApertureSize = 3;
const double kFreeParam = 0.04;

constexpr char kQuitKey = 'q';

constexpr char kHarrisCorner[] = "Harris Corner Detection";
constexpr char kThreshold[] = "Threshold";

int main() {
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

  int threshold = 150; // Initial threshold for corner detection

  // Create a window to display the results and trackbars to adjust parameters
  cv::namedWindow(kHarrisCorner, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar(kThreshold, kHarrisCorner, &threshold, 255);

  while (true) {
    cv::Mat frame, gray, dst, dst_norm, dst_norm_scaled;

    camera >> frame;
    if (frame.empty()) {
      LOG_ERROR("Error: No frame captured.");
      break;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Harris corner detection
    dst = cv::Mat::zeros(gray.size(), CV_32FC1);
    cv::cornerHarris(gray, dst, kBlockSize, kApertureSize, kFreeParam);

    // Normalize and convert to 8-bit image
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Draw corners on the original frame
    for (int y = 0; y < dst_norm.rows; y++) {
      for (int x = 0; x < dst_norm.cols; x++) {
        if ((int)dst_norm.at<float>(y, x) > threshold) {
          cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), 2,
                     cv::LINE_AA);
        }
      }
    }

    cv::imshow(kHarrisCorner, frame);

    char new_key = cv::waitKey(10);
    if (new_key >= 0) {
      if (kQuitKey == new_key) {
        break;
      }
    }
  }

  camera.release();
  cv::destroyAllWindows();
  return 0;
}
