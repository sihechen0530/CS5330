#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <opencv2/opencv.hpp>

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

#define ASSERT_EQ(A, B)                                                        \
  do {                                                                         \
    if (A != B) {                                                              \
      LOG_ERROR("assert equal failed: " << A << " " << B);                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

cv::Mat image_reader(const std::string &image_path);

int showMatches(const std::string &directory,
                const std::vector<std::string> &matches);

#endif
