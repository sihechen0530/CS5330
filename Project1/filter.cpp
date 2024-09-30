/*
  Sihe Chen (002085773)
  Fall 2024
  CS 5330 Project 1
  source file for multiple filters
*/
#include "filter.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>

int convertGreyScale(const cv::Mat &src, cv::Mat &dst) {
  cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
  return 0;
}

int convertGreyScale2(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), CV_8UC1);
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      const auto &src_pixel = src.at<cv::Vec3b>(i, j);
      dst.at<uint8_t>(i, j) =
          std::max(std::max(src_pixel[0], src_pixel[1]), src_pixel[2]);
    }
  }
  return 0;
}

int convertSepiaTone(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  auto compute = [](const cv::Vec3b &src_pixel, const float red_coeff,
                    const float green_coeff,
                    const float blue_coeff) -> uint8_t {
    // src pixel in the order of B G R
    float weighted_sum = static_cast<float>(src_pixel[0]) * blue_coeff +
                         static_cast<float>(src_pixel[1]) * green_coeff +
                         static_cast<float>(src_pixel[2]) * red_coeff;
    uint8_t ret = static_cast<uint8_t>(
        std::min(weighted_sum, static_cast<float>(UINT8_MAX)));
    return ret;
  };
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      const auto &src_pixel = src.at<cv::Vec3b>(i, j);
      auto &dst_pixel = dst.at<cv::Vec3b>(i, j);
      // blue
      dst_pixel[0] = compute(src_pixel, 0.272, 0.534, 0.131);
      // green
      dst_pixel[1] = compute(src_pixel, 0.349, 0.686, 0.168);
      // red
      dst_pixel[2] = compute(src_pixel, 0.393, 0.769, 0.189);
    }
  }
  return 0;
}

int blur5x5_1(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  // initialize coefficient matrix
  uint8_t split[] = {1, 2, 4, 2, 1};
  uint8_t coeff[5][5] = {0};
  int sum = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      coeff[i][j] = split[i] * split[j];
      sum += coeff[i][j];
    }
  }
  // blur
  for (int i = 2; i < src.rows - 2; ++i) {
    for (int j = 2; j < src.cols - 2; ++j) {
      for (int c = 0; c < 3; ++c) {
        uint32_t conv = 0;
        for (int k = -2; k <= 2; ++k) {
          for (int l = -2; l <= 2; ++l) {
            conv += src.at<cv::Vec3b>(i + k, j + l)[c] * coeff[k + 2][l + 2];
          }
        }
        dst.at<cv::Vec3b>(i, j)[c] = static_cast<uint8_t>(conv / sum);
      }
    }
  }
  return 0;
}

int blur5x5_2(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  // initialize coefficient matrix
  int split[] = {1, 2, 4, 2, 1};
  int coeff[5][5] = {0};
  int sum = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      coeff[i][j] = split[i] * split[j];
      sum += coeff[i][j];
    }
  }
  // blur
  cv::Mat mul_row(src.rows, src.cols, CV_32SC3);
  cv::Mat mul_col(src.rows, src.cols, CV_32SC3);
  // 1. compute rows
  for (int i = 2; i < src.rows - 2; ++i) {
    const cv::Vec3b *src_m2_ptr = src.ptr<cv::Vec3b>(i - 2);
    const cv::Vec3b *src_m1_ptr = src.ptr<cv::Vec3b>(i - 1);
    const cv::Vec3b *src_0_ptr = src.ptr<cv::Vec3b>(i);
    const cv::Vec3b *src_p1_ptr = src.ptr<cv::Vec3b>(i + 1);
    const cv::Vec3b *src_p2_ptr = src.ptr<cv::Vec3b>(i + 2);
    cv::Vec3i *mul_row_ptr = mul_row.ptr<cv::Vec3i>(i);
    for (int j = 0; j < src.cols; ++j) {
      mul_row_ptr[j][0] = static_cast<int>(src_m2_ptr[j][0]) * split[0] +
                          static_cast<int>(src_m1_ptr[j][0]) * split[1] +
                          static_cast<int>(src_0_ptr[j][0]) * split[2] +
                          static_cast<int>(src_p1_ptr[j][0]) * split[3] +
                          static_cast<int>(src_p2_ptr[j][0]) * split[4];
      mul_row_ptr[j][1] = static_cast<int>(src_m2_ptr[j][1]) * split[0] +
                          static_cast<int>(src_m1_ptr[j][1]) * split[1] +
                          static_cast<int>(src_0_ptr[j][1]) * split[2] +
                          static_cast<int>(src_p1_ptr[j][1]) * split[3] +
                          static_cast<int>(src_p2_ptr[j][1]) * split[4];
      mul_row_ptr[j][2] = static_cast<int>(src_m2_ptr[j][2]) * split[0] +
                          static_cast<int>(src_m1_ptr[j][2]) * split[1] +
                          static_cast<int>(src_0_ptr[j][2]) * split[2] +
                          static_cast<int>(src_p1_ptr[j][2]) * split[3] +
                          static_cast<int>(src_p2_ptr[j][2]) * split[4];
    }
  }
  // 2. compute cols
  for (int i = 0; i < src.rows; ++i) {
    const cv::Vec3i *mul_row_ptr = mul_row.ptr<cv::Vec3i>(i);
    cv::Vec3i *mul_col_ptr = mul_col.ptr<cv::Vec3i>(i);
    for (int j = 2; j < src.cols - 2; ++j) {
      mul_col_ptr[j][0] =
          mul_row_ptr[j - 2][0] * split[0] + mul_row_ptr[j - 1][0] * split[1] +
          mul_row_ptr[j][0] * split[2] + mul_row_ptr[j + 1][0] * split[3] +
          mul_row_ptr[j + 2][0] * split[4];
      mul_col_ptr[j][1] =
          mul_row_ptr[j - 2][1] * split[0] + mul_row_ptr[j - 1][1] * split[1] +
          mul_row_ptr[j][1] * split[2] + mul_row_ptr[j + 1][1] * split[3] +
          mul_row_ptr[j + 2][1] * split[4];
      mul_col_ptr[j][2] =
          mul_row_ptr[j - 2][2] * split[0] + mul_row_ptr[j - 1][2] * split[1] +
          mul_row_ptr[j][2] * split[2] + mul_row_ptr[j + 1][2] * split[3] +
          mul_row_ptr[j + 2][2] * split[4];
    }
  }
  // 3. fill result
  for (int i = 0; i < src.rows; ++i) {
    const cv::Vec3i *mul_col_ptr = mul_col.ptr<cv::Vec3i>(i);
    cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; ++j) {
      dst_ptr[j][0] = static_cast<uint8_t>(mul_col_ptr[j][0] / sum);
      dst_ptr[j][1] = static_cast<uint8_t>(mul_col_ptr[j][1] / sum);
      dst_ptr[j][2] = static_cast<uint8_t>(mul_col_ptr[j][2] / sum);
    }
  }

  // // loop unrolling + pointers
  // for (int i = 2; i < src.rows - 2; ++i) {
  //   const cv::Vec3b *row_m2 = src.ptr<cv::Vec3b>(i - 2);
  //   const cv::Vec3b *row_m1 = src.ptr<cv::Vec3b>(i - 1);
  //   const cv::Vec3b *row = src.ptr<cv::Vec3b>(i);
  //   const cv::Vec3b *row_p1 = src.ptr<cv::Vec3b>(i + 1);
  //   const cv::Vec3b *row_p2 = src.ptr<cv::Vec3b>(i + 2);
  //   cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);
  //   for (int j = 2; j < src.cols - 2; ++j) {
  //     uint32_t conv = 0;
  //     conv += row_m2[j - 2][0] * coeff[0][0] + row_m2[j - 1][0] * coeff[0][1]
  //     +
  //             row_m2[j][0] * coeff[0][2] + row_m2[j + 1][0] * coeff[0][3] +
  //             row_m2[j + 2][0] * coeff[0][4] + row_m1[j - 2][0] * coeff[1][0]
  //             + row_m1[j - 1][0] * coeff[1][1] + row_m1[j][0] * coeff[1][2] +
  //             row_m1[j + 1][0] * coeff[1][3] + row_m1[j + 2][0] * coeff[1][4]
  //             + row[j - 2][0] * coeff[2][0] + row[j - 1][0] * coeff[2][1] +
  //             row[j][0] * coeff[2][2] + row[j + 1][0] * coeff[2][3] +
  //             row[j + 2][0] * coeff[2][4] + row_p1[j - 2][0] * coeff[3][0] +
  //             row_p1[j - 1][0] * coeff[3][1] + row_p1[j][0] * coeff[3][2] +
  //             row_p1[j + 1][0] * coeff[3][3] + row_p1[j + 2][0] * coeff[3][4]
  //             + row_p2[j - 2][0] * coeff[4][0] + row_p2[j - 1][0] *
  //             coeff[4][1] + row_p2[j][0] * coeff[4][2] + row_p2[j + 1][0] *
  //             coeff[4][3] + row_p2[j + 2][0] * coeff[4][4];
  //     dst_ptr[j][0] = static_cast<uint8_t>(conv / sum);
  //     conv = 0;
  //     conv += row_m2[j - 2][1] * coeff[0][0] + row_m2[j - 1][1] * coeff[0][1]
  //     +
  //             row_m2[j][1] * coeff[0][2] + row_m2[j + 1][1] * coeff[0][3] +
  //             row_m2[j + 2][1] * coeff[0][4] + row_m1[j - 2][1] * coeff[1][0]
  //             + row_m1[j - 1][1] * coeff[1][1] + row_m1[j][1] * coeff[1][2] +
  //             row_m1[j + 1][1] * coeff[1][3] + row_m1[j + 2][1] * coeff[1][4]
  //             + row[j - 2][1] * coeff[2][0] + row[j - 1][1] * coeff[2][1] +
  //             row[j][1] * coeff[2][2] + row[j + 1][1] * coeff[2][3] +
  //             row[j + 2][1] * coeff[2][4] + row_p1[j - 2][1] * coeff[3][0] +
  //             row_p1[j - 1][1] * coeff[3][1] + row_p1[j][1] * coeff[3][2] +
  //             row_p1[j + 1][1] * coeff[3][3] + row_p1[j + 2][1] * coeff[3][4]
  //             + row_p2[j - 2][1] * coeff[4][0] + row_p2[j - 1][1] *
  //             coeff[4][1] + row_p2[j][1] * coeff[4][2] + row_p2[j + 1][1] *
  //             coeff[4][3] + row_p2[j + 2][1] * coeff[4][4];
  //     dst_ptr[j][1] = static_cast<uint8_t>(conv / sum);
  //     conv = 0;
  //     conv += row_m2[j - 2][2] * coeff[0][0] + row_m2[j - 1][2] * coeff[0][1]
  //     +
  //             row_m2[j][2] * coeff[0][2] + row_m2[j + 1][2] * coeff[0][3] +
  //             row_m2[j + 2][2] * coeff[0][4] + row_m1[j - 2][2] * coeff[1][0]
  //             + row_m1[j - 1][2] * coeff[1][1] + row_m1[j][2] * coeff[1][2] +
  //             row_m1[j + 1][2] * coeff[1][3] + row_m1[j + 2][2] * coeff[1][4]
  //             + row[j - 2][2] * coeff[2][0] + row[j - 1][2] * coeff[2][1] +
  //             row[j][2] * coeff[2][2] + row[j + 1][2] * coeff[2][3] +
  //             row[j + 2][2] * coeff[2][4] + row_p1[j - 2][2] * coeff[3][0] +
  //             row_p1[j - 1][2] * coeff[3][1] + row_p1[j][2] * coeff[3][2] +
  //             row_p1[j + 1][2] * coeff[3][3] + row_p1[j + 2][2] * coeff[3][4]
  //             + row_p2[j - 2][2] * coeff[4][0] + row_p2[j - 1][2] *
  //             coeff[4][1] + row_p2[j][2] * coeff[4][2] + row_p2[j + 1][2] *
  //             coeff[4][3] + row_p2[j + 2][2] * coeff[4][4];
  //     dst_ptr[j][2] = static_cast<uint8_t>(conv / sum);
  //   }
  // }
  return 0;
}

// util function. not declared in header file as user won't call this function.
// computational helper function since a majority of code is the same in sobelX
// and sobelY implementation.
int sobel3x3(const int *row_coeff, const int *col_coeff, const int sum,
             const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), CV_16SC3);
  // save intermediate result
  cv::Mat mul_row(src.size(), CV_32SC3);
  cv::Mat mul_col(src.size(), CV_32SC3);
  // row multiplication
  for (int i = 1; i < src.rows - 1; ++i) {
    const cv::Vec3b *src_m1_ptr = src.ptr<cv::Vec3b>(i - 1);
    const cv::Vec3b *src_0_ptr = src.ptr<cv::Vec3b>(i);
    const cv::Vec3b *src_p1_ptr = src.ptr<cv::Vec3b>(i + 1);
    cv::Vec3i *mul_row_ptr = mul_row.ptr<cv::Vec3i>(i);
    for (int j = 0; j < src.cols; ++j) {
      mul_row_ptr[j][0] = static_cast<int>(src_m1_ptr[j][0]) * row_coeff[0] +
                          static_cast<int>(src_0_ptr[j][0]) * row_coeff[1] +
                          static_cast<int>(src_p1_ptr[j][0]) * row_coeff[2];
      mul_row_ptr[j][1] = static_cast<int>(src_m1_ptr[j][1]) * row_coeff[0] +
                          static_cast<int>(src_0_ptr[j][1]) * row_coeff[1] +
                          static_cast<int>(src_p1_ptr[j][1]) * row_coeff[2];
      mul_row_ptr[j][2] = static_cast<int>(src_m1_ptr[j][2]) * row_coeff[0] +
                          static_cast<int>(src_0_ptr[j][2]) * row_coeff[1] +
                          static_cast<int>(src_p1_ptr[j][2]) * row_coeff[2];
    }
  }
  // col multiplication
  for (int i = 0; i < src.rows; ++i) {
    const cv::Vec3i *mul_row_ptr = mul_row.ptr<cv::Vec3i>(i);
    cv::Vec3s *dst_ptr = dst.ptr<cv::Vec3s>(i);
    for (int j = 1; j < src.cols - 1; ++j) {
      int conv = mul_row_ptr[j - 1][0] * col_coeff[0] +
                 mul_row_ptr[j][0] * col_coeff[1] +
                 mul_row_ptr[j + 1][0] * col_coeff[2];
      dst_ptr[j][0] = static_cast<int16_t>(conv / sum);
      conv = mul_row_ptr[j - 1][1] * col_coeff[0] +
             mul_row_ptr[j][1] * col_coeff[1] +
             mul_row_ptr[j + 1][1] * col_coeff[2];
      dst_ptr[j][1] = static_cast<int16_t>(conv / sum);
      conv = mul_row_ptr[j - 1][2] * col_coeff[0] +
             mul_row_ptr[j][2] * col_coeff[1] +
             mul_row_ptr[j + 1][2] * col_coeff[2];
      dst_ptr[j][2] = static_cast<int16_t>(conv / sum);
    }
  }
  return 0;
}

int sobelX3x3(const cv::Mat &src, cv::Mat &sobel_result) {
  constexpr int row_coeff[3] = {-1, 0, 1};
  constexpr int col_coeff[3] = {1, 2, 1};
  constexpr int sum = 4;
  return sobel3x3(row_coeff, col_coeff, sum, src, sobel_result);
}

int sobelY3x3(const cv::Mat &src, cv::Mat &sobel_result) {
  constexpr int row_coeff[3] = {1, 2, 1};
  constexpr int col_coeff[3] = {-1, 0, 1};
  constexpr int sum = 4;
  return sobel3x3(row_coeff, col_coeff, sum, src, sobel_result);
}

int sobelVisualize(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), CV_8UC3);
  // [-255, 255] -> [0, 255]
  constexpr double scale = 0.5f;
  constexpr double bias = 128.0f;
  cv::convertScaleAbs(src, dst, scale, bias);
  return 0;
}

int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst) {
  if (sx.size() != sy.size()) {
    std::cerr << "Matrix size not same!" << std::endl;
    return 1;
  }
  dst.create(sx.size(), CV_8UC3);
  for (int i = 0; i < sx.rows; ++i) {
    const cv::Vec3s *sx_ptr = sx.ptr<cv::Vec3s>(i);
    const cv::Vec3s *sy_ptr = sy.ptr<cv::Vec3s>(i);
    cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < sx.cols; ++j) {
      dst_ptr[j][0] = static_cast<uint8_t>(
          std::sqrt(sx_ptr[j][0] * sx_ptr[j][0] + sy_ptr[j][0] * sy_ptr[j][0]));
      dst_ptr[j][1] = static_cast<uint8_t>(
          std::sqrt(sx_ptr[j][1] * sx_ptr[j][1] + sy_ptr[j][1] * sy_ptr[j][1]));
      dst_ptr[j][2] = static_cast<uint8_t>(
          std::sqrt(sx_ptr[j][2] * sx_ptr[j][2] + sy_ptr[j][2] * sy_ptr[j][2]));
    }
  }
  return 0;
}

int blurQuantize(const cv::Mat &src, cv::Mat &dst, const int levels) {
  dst.create(src.size(), CV_8UC3);
  const int bucket = 255 / levels;
  cv::Mat blurred_img;
  blur5x5_2(src, blurred_img);
  for (int i = 0; i < blurred_img.rows; ++i) {
    cv::Vec3b *img_ptr = blurred_img.ptr<cv::Vec3b>(i);
    cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < blurred_img.cols; ++j) {
      dst_ptr[j][0] = static_cast<uint8_t>(img_ptr[j][0] / bucket) * bucket;
      dst_ptr[j][1] = static_cast<uint8_t>(img_ptr[j][1] / bucket) * bucket;
      dst_ptr[j][2] = static_cast<uint8_t>(img_ptr[j][2] / bucket) * bucket;
    }
  }
  return 0;
}

int glassDistort(const cv::Mat &src, const int distortionStrength,
                 cv::Mat &dst) {
  dst = cv::Mat::zeros(src.size(), src.type());
  int h = src.rows;
  int w = src.cols;

  // use fixed random seed so that the result is consistent
  std::srand(1234);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int offsetX = std::rand() % (2 * distortionStrength) - distortionStrength;
      int offsetY = std::rand() % (2 * distortionStrength) - distortionStrength;

      int newX = std::clamp(x + offsetX, 0, w - 1);
      int newY = std::clamp(y + offsetY, 0, h - 1);

      dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(newY, newX);
    }
  }

  return 0;
}

int TimeDelay::timeDelay(const cv::Mat &src, cv::Mat &dst) {
  frames_.push_back(src);
  if (frames_.size() > count_) {
    frames_.pop_front();
  }
  dst = cv::Mat::zeros(src.size(), src.type());

  for (const auto &frame : frames_) {
    // apply the current frame with weight to dst
    dst = dst * (1.0 - alpha_) + frame * alpha_;
  }
  return 0;
}

int brighten(const cv::Mat &src, const int step, cv::Mat &dst) {
  cv::Mat hsvImage;
  cv::cvtColor(src, hsvImage, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> hsvChannels;
  cv::split(hsvImage, hsvChannels);

  hsvChannels[2] += step;

  cv::merge(hsvChannels, hsvImage);

  dst = cv::Mat(src.size(), src.type());
  cv::cvtColor(hsvImage, dst, cv::COLOR_HSV2BGR);
  return 0;
}

int improveFace(const cv::Mat &src, const std::vector<cv::Rect> &faces,
                cv::Mat &dst) {
  // Step 0: set up the result image
  src.copyTo(dst);
  // Step 1: get edges
  cv::Mat sobel_x_res;
  sobelX3x3(src, sobel_x_res);
  cv::Mat sobel_y_res;
  sobelY3x3(src, sobel_y_res);
  cv::Mat sobel_res;
  magnitude(sobel_x_res, sobel_y_res, sobel_res);
  cv::Mat sobel_grey;
  convertGreyScale2(sobel_res, sobel_grey);
  // Step 2: blur the whole image
  cv::Mat blurred_img;
  blur5x5_2(src, blurred_img);
  // Step 3: brighten
  cv::Mat brightened;
  brighten(src, 50, brightened);
  // Step 4: copy the blurred face except edge to the original image
  for (const auto &face : faces) {
    for (int i = std::max(0, face.y - face.height / 2);
         i < std::min(src.rows, face.y + face.height * 3 / 2); ++i) {
      for (int j = std::max(0, face.x - face.width / 2);
           j < std::min(src.cols, face.x + face.width * 3 / 2); ++j) {
        if (sobel_grey.at<uint8_t>(i, j) < 20) {
          // is not edge, use blur
          dst.at<cv::Vec3b>(i, j) = brightened.at<cv::Vec3b>(i, j);
        }
      }
    }
  }
  return 0;
}