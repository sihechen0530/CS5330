/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * source file for util functions
 */

#include "utils.h"
#include <cmath>

int showImage(const std::string &title, const cv::Mat &frame) {
  // cv::Mat show;
  // cv::resize(frame, show, cv::Size(), 1, 1);
  cv::imshow(title, frame);
  return 0;
}

bool isPointInRotatedRect(const cv::Point2f &point,
                          const cv::RotatedRect &rectangle) {
  cv::Point2f corners[4];
  rectangle.points(corners);
  cv::Point2f *lastItemPointer = (corners + sizeof corners / sizeof corners[0]);
  std::vector<cv::Point2f> contour(corners, lastItemPointer);
  double indicator = cv::pointPolygonTest(contour, point, false);
  return indicator >= 0;
}
