/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * header file for util functions
 */

#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define LOG_INFO(x)                                                            \
  do {                                                                         \
    std::cout << "[INFO] " << x << std::endl;                                  \
  } while (0)

#define LOG_WARN(x)                                                            \
  do {                                                                         \
    std::cout << "[WARN] " << x << std::endl;                                  \
  } while (0)

#define LOG_ERROR(x)                                                           \
  do {                                                                         \
    std::cerr << "[ERROR] " << x << std::endl;                                 \
  } while (0)

#define ASSERT_EQ_RET(A, B, C)                                                 \
  do {                                                                         \
    if (A != B) {                                                              \
      LOG_ERROR("assert equal failed: " << A << " " << B);                     \
      return C;                                                                \
    }                                                                          \
  } while (0)

#define GET_INPUT_WITH_PROMPT(PROMPT, INPUT)                                   \
  do {                                                                         \
    LOG_INFO(PROMPT);                                                          \
    std::cin >> INPUT;                                                         \
  } while (0)

int showImage(const std::string &title, const cv::Mat &frame);

bool isPointInRotatedRect(const cv::Point2f &point,
                          const cv::RotatedRect &rRect);
#endif
