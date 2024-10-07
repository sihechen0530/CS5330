#include "feature_extractor.h"
#include "utils.h"
#include <functional>
#include <memory>
#include <string>

#define CHECK_IMAGE(image)                                                     \
  do {                                                                         \
    if (!image.data) {                                                         \
      LOG_ERROR("no image data");                                              \
      return -1;                                                               \
    }                                                                          \
    if (CV_8UC3 != image.type()) {                                             \
      LOG_ERROR("unsupported image type");                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

const std::string kFeatureExtractorKey = "extractor";
const std::string kSplitFuncKey = "region";
const std::string kConfigKey = "config";
const std::string kBinSizeKey = "bin_size";
const std::string kSigmaKey = "sigma";

const std::unordered_map<
    std::string, std::function<int(const cv::Mat &, const nlohmann::json &,
                                   std::vector<float> *)>>
    kFeatureExtractors = {{"roi", roi},
                          {"rgHistogram", rgHistogram},
                          {"rgbHistogram", rgbHistogram},
                          {"sobelHistogram", sobelHistogram},
                          {"laws", lawsFilterHistogram},
                          {"gabor", gaborFilterHistogram},
                          {"fourier", fourierTransform}};

const std::unordered_map<std::string, std::function<cv::Mat(const cv::Mat &)>>
    kSplitFuncs = {{"whole", getWholeImage},       {"upper", getUpperHalf},
                   {"lower", getLowerHalf},        {"left", getLeftHalf},
                   {"right", getRightHalf},        {"upper_left", getUpperLeft},
                   {"upper_right", getUpperRight}, {"lower_left", getLowerLeft},
                   {"lower_right", getLowerRight}};

cv::Mat getWholeImage(const cv::Mat &src) { return src; }

cv::Mat getUpperHalf(const cv::Mat &src) {
  cv::Rect upper_half(0, 0, src.cols, src.rows / 2);
  return src(upper_half).clone();
}

cv::Mat getLowerHalf(const cv::Mat &src) {
  cv::Rect lower_half(0, src.rows / 2, src.cols, src.rows / 2);
  return src(lower_half).clone();
}

cv::Mat getLeftHalf(const cv::Mat &src) {
  cv::Rect left_half(0, 0, src.cols / 2, src.rows);
  return src(left_half).clone();
}

cv::Mat getRightHalf(const cv::Mat &src) {
  cv::Rect right_half(src.cols / 2, 0, src.cols / 2, src.rows);
  return src(right_half).clone();
}

cv::Mat getUpperLeft(const cv::Mat &src) {
  cv::Rect upper_left(0, 0, src.cols / 2, src.rows / 2);
  return src(upper_left).clone();
}

cv::Mat getUpperRight(const cv::Mat &src) {
  cv::Rect upper_right(src.cols / 2, 0, src.cols / 2, src.rows / 2);
  return src(upper_right).clone();
}

cv::Mat getLowerLeft(const cv::Mat &src) {
  cv::Rect lower_left(0, src.rows / 2, src.cols / 2, src.rows / 2);
  return src(lower_left).clone();
}

cv::Mat getLowerRight(const cv::Mat &src) {
  cv::Rect lower_right(src.cols / 2, src.rows / 2, src.cols / 2, src.rows / 2);
  return src(lower_right).clone();
}

int roi(const cv::Mat &m, const nlohmann::json &config,
        std::vector<float> *feature) {
  // step 1: get roi
  const int kRoiSize = config["roi_size"];
  cv::Rect roi(m.cols / 2 - kRoiSize / 2, m.rows / 2 - kRoiSize / 2, kRoiSize,
               kRoiSize);
  // step 2: extract roi feature
  feature->reserve(roi.area() * 3);
  for (int i = std::max(0, roi.y); i < std::min(m.rows, roi.y + roi.height);
       ++i) {
    const cv::Vec3b *ptr = m.ptr<cv::Vec3b>(i);
    for (int j = std::max(0, roi.x); j < std::min(m.cols, roi.x + roi.width);
         ++j) {
      feature->emplace_back(static_cast<float>(ptr[j][0]));
      feature->emplace_back(static_cast<float>(ptr[j][1]));
      feature->emplace_back(static_cast<float>(ptr[j][2]));
    }
  }
  return 0;
}

int rgHistogram(const cv::Mat &m, const nlohmann::json &config,
                std::vector<float> *feature) {
  // step 1: get rg chromaticity space histogram
  const int kRGBinSize = config[kBinSizeKey];
  // use gaussian noise for soft histogram
  float sigma = 0.0f;
  if (config.contains(kSigmaKey)) {
    sigma = config[kSigmaKey];
  }
  auto gaussian = [&sigma](const int x) -> float {
    return std::exp(-0.5 * (x * x) / (sigma * sigma)) /
           (sigma * std::sqrt(2 * CV_PI));
  };
  feature->resize(kRGBinSize * 2, 0.0f);
  for (int i = 0; i < m.rows; ++i) {
    const cv::Vec3b *ptr = m.ptr<cv::Vec3b>(i);
    for (int j = 0; j < m.cols; j++) {
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      float divisor = R + G + B;
      divisor = divisor > 0.0 ? divisor : 1.0;

      float r = R / divisor;
      float g = G / divisor;

      int rindex = (int)(r * (kRGBinSize - 1) + 0.5);
      int gindex = (int)(g * (kRGBinSize - 1) + 0.5);

      if (0.0f == sigma) {
        ++(*feature)[rindex];
        ++(*feature)[kRGBinSize + gindex];
      } else {
        // soft histogram
        for (int k = 0; k < kRGBinSize; ++k) {
          (*feature)[rindex] += gaussian(std::abs(rindex - k));
          (*feature)[kRGBinSize + gindex] += gaussian(std::abs(gindex - k));
        }
      }
    }
  }
  // step 2: normalize
  for (auto &feat : *feature) {
    feat /= (m.rows * m.cols);
  }
  return 0;
}

int rgbHistogram(const cv::Mat &m, const nlohmann::json &config,
                 std::vector<float> *feature) {
  // step 1: get rg chromaticity space histogram
  const int kRGBBinSize = config[kBinSizeKey];
  const float divisor = 256.0f;
  feature->resize(kRGBBinSize * 3, 0.0f);
  for (int i = 0; i < m.rows; ++i) {
    const cv::Vec3b *ptr = m.ptr<cv::Vec3b>(i);
    for (int j = 0; j < m.cols; j++) {
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      int rindex = (int)(R / divisor * (kRGBBinSize - 1) + 0.5);
      int gindex = (int)(G / divisor * (kRGBBinSize - 1) + 0.5);
      int bindex = (int)(B / divisor * (kRGBBinSize - 1) + 0.5);

      ++(*feature)[rindex];
      ++(*feature)[kRGBBinSize + gindex];
      ++(*feature)[kRGBBinSize * 2 + bindex];
    }
  }
  // step 2: normalize
  for (auto &feat : *feature) {
    feat /= (m.rows * m.cols);
  }
  return 0;
}

int sobel3x3(const int *row_coeff, const int *col_coeff, const int sum,
             const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), CV_64FC1);
  // save intermediate result
  cv::Mat mul_row(src.size(), CV_64FC1);
  cv::Mat mul_col(src.size(), CV_64FC1);
  // row multiplication
  for (int i = 1; i < src.rows - 1; ++i) {
    const uint8_t *src_m1_ptr = src.ptr<uint8_t>(i - 1);
    const uint8_t *src_0_ptr = src.ptr<uint8_t>(i);
    const uint8_t *src_p1_ptr = src.ptr<uint8_t>(i + 1);
    double *mul_row_ptr = mul_row.ptr<double>(i);
    for (int j = 0; j < src.cols; ++j) {
      mul_row_ptr[j] = src_m1_ptr[j] * row_coeff[0] +
                       src_0_ptr[j] * row_coeff[1] +
                       src_p1_ptr[j] * row_coeff[2];
    }
  }
  // col multiplication
  for (int i = 0; i < src.rows; ++i) {
    const double *mul_row_ptr = mul_row.ptr<double>(i);
    double *dst_ptr = dst.ptr<double>(i);
    for (int j = 1; j < src.cols - 1; ++j) {
      double conv = mul_row_ptr[j - 1] * col_coeff[0] +
                    mul_row_ptr[j] * col_coeff[1] +
                    mul_row_ptr[j + 1] * col_coeff[2];
      dst_ptr[j] = conv / sum;
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

int cartToPolar(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &magnitude,
                cv::Mat &angle) {
  if (sx.size() != sy.size()) {
    std::cerr << "Matrix size not same!" << std::endl;
    return 1;
  }
  magnitude.create(sx.size(), CV_64FC1);
  angle.create(sx.size(), CV_64FC1);
  for (int i = 0; i < sx.rows; ++i) {
    const double *sx_ptr = sx.ptr<double>(i);
    const double *sy_ptr = sy.ptr<double>(i);
    double *magnitude_ptr = magnitude.ptr<double>(i);
    double *angle_ptr = angle.ptr<double>(i);
    for (int j = 0; j < sx.cols; ++j) {
      magnitude_ptr[j] =
          std::sqrt(sx_ptr[j] * sx_ptr[j] + sy_ptr[j] * sy_ptr[j]);
      double angle = std::atan2(sy_ptr[j], sx_ptr[j]);
      if (angle < 0)
        angle += 2 * CV_PI;
      angle_ptr[j] = angle;
    }
  }
  return 0;
}

int sobelHistogram(const cv::Mat &m, const nlohmann::json &config,
                   std::vector<float> *feature) {
  const int kBinSize = config[kBinSizeKey];
  cv::Mat gray;
  cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
  cv::Mat grad_x, grad_y, magnitude, angle;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3, 0.25);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3, 0.25);
  cv::cartToPolar(grad_x, grad_y, magnitude, angle);
  float magnitude_range[] = {0, 256}, angle_range[] = {0, 2 * CV_PI};
  const float *magnitudeHistRange = {magnitude_range},
              *angleHistRange = {angle_range};
  int histSize = kBinSize;
  cv::Mat hist;
  cv::calcHist(&magnitude, 1, 0, cv::Mat(), hist, 1, &histSize,
               &magnitudeHistRange);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  feature->assign((float *)hist.data,
                  (float *)hist.data + hist.rows * hist.cols);
  cv::calcHist(&magnitude, 1, 0, cv::Mat(), hist, 1, &histSize,
               &angleHistRange);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  feature->insert(feature->end(), (float *)hist.data,
                  (float *)hist.data + hist.rows * hist.cols);

  // calculate both magnitude and direction
  // sobelX3x3(gray, grad_x);
  // sobelY3x3(gray, grad_y);
  // cartToPolar(grad_x, grad_y, magnitude, angle);
  // ASSERT_EQ(magnitude.size(), angle.size());
  // const float magnitude_divisor = 256.0f;
  // const float angle_divisor = 2 * CV_PI;
  // for (int i = 0; i < magnitude.rows; ++i) {
  //   double *mptr = magnitude.ptr<double>(i);
  //   double *aptr = angle.ptr<double>(i);
  //   for (int j = 0; j < magnitude.cols; ++j) {
  //     int mindex = (int)(mptr[j] / magnitude_divisor * (kBinSize - 1) + 0.5);
  //     int aindex = (int)(aptr[j] / angle_divisor * (kBinSize - 1) + 0.5);
  //     if (mindex >= 0 && mindex < kBinSize) {
  //       ++(*feature)[mindex];
  //     } else {
  //       LOG_ERROR("index out of bucket range: " << mindex << " " << mptr[j]);
  //     }
  //     if (aindex >= 0 && aindex < kBinSize) {
  //       ++(*feature)[kBinSize + aindex];
  //     } else {
  //       LOG_ERROR("index out of bucket range: " << aindex << " " << aptr[j]);
  //     }
  //   }
  // }
  // step 2: normalize
  // for (auto &feat : *feature) {
  //   feat /= (m.rows * m.cols);
  // }
  return 0;
}

int lawsFilterHistogram(const cv::Mat &m, const nlohmann::json &config,
                        std::vector<float> *feature) {
  int kBinSize = config["bin_size"];
  float range[] = {0, 256};
  const float *histRange = {range};
  feature->clear();
  feature->reserve(kBinSize * 5);

  cv::Mat gray;
  cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Mat> kernels = {
      (cv::Mat_<float>(1, 5) << -1, -2, 0, 2, 1), // L5
      (cv::Mat_<float>(1, 5) << -1, 0, 2, 0, -1), // E5
      (cv::Mat_<float>(1, 5) << -1, 2, 0, -2, 1), // S5
      (cv::Mat_<float>(1, 5) << 1, -4, 6, -4, 1), // R5
      (cv::Mat_<float>(1, 5) << -1, 4, -6, 4, -1) // W5
  };

  for (const auto &kernel : kernels) {
    cv::Mat response;
    cv::filter2D(gray, response, CV_32F, kernel);
    cv::Mat hist;
    cv::calcHist(&response, 1, 0, cv::Mat(), hist, 1, &kBinSize, &histRange);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    feature->insert(feature->end(), (float *)hist.data,
                    (float *)hist.data + hist.rows * hist.cols);
  }
  return 0;
}

int gaborFilterHistogram(const cv::Mat &m, const nlohmann::json &config,
                         std::vector<float> *feature) {
  int kBinSize = config["bin_size"];
  float range[] = {0, 256};
  const float *histRange = {range};
  feature->clear();
  feature->reserve(config["frequencies"].size() *
                   config["orientations"].size() * kBinSize);

  cv::Mat gray;
  cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);

  for (float frequency : config["frequencies"]) {
    for (float orientation : config["orientations"]) {
      cv::Mat kernel = cv::getGaborKernel(cv::Size(31, 31), 4.0, orientation,
                                          frequency, 0.5, 0, CV_32F);
      cv::Mat response;
      cv::filter2D(gray, response, CV_32F, kernel);
      cv::Mat hist;
      cv::calcHist(&response, 1, 0, cv::Mat(), hist, 1, &kBinSize, &histRange);
      cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
      feature->insert(feature->end(), (float *)hist.data,
                      (float *)hist.data + hist.rows * hist.cols);
    }
  }
  return 0;
}

int fourierTransform(const cv::Mat &mat, const nlohmann::json &config,
                     std::vector<float> *feature) {
  int kResize = config["resize"];
  cv::Mat gray;
  cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);

  // Expand input image to optimal size
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(gray.rows);
  int n = cv::getOptimalDFTSize(gray.cols);
  cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Create a complex matrix to hold the DFT coefficients
  cv::Mat planes[] = {cv::Mat_<float>(padded),
                      cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);

  // Perform the DFT
  cv::dft(complexI, complexI);

  // Compute the magnitude of the DFT coefficients
  cv::split(complexI, planes);
  cv::magnitude(planes[0], planes[1], planes[0]);
  cv::Mat magI = planes[0];

  // Switch to logarithmic scale
  magI += cv::Scalar::all(1);
  cv::log(magI, magI);

  // Crop the spectrum, if it has an odd number of rows or columns
  magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

  // Rearrange the quadrants of the Fourier image so that the origin is at the
  // center
  int cx = magI.cols / 2;
  int cy = magI.rows / 2;

  cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
  cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // Normalize the magnitude image
  cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

  // Resize to 16x16
  cv::Mat resizedMagI;
  cv::resize(magI, resizedMagI, cv::Size(kResize, kResize));

  feature->clear();
  feature->assign((float *)resizedMagI.data,
                  (float *)resizedMagI.data +
                      resizedMagI.rows * resizedMagI.cols);

  return 0;
}

int extract(const std::string &image_path, const nlohmann::json &extract_config,
            std::vector<float> *feature) {
  // step 1: load image from file
  cv::Mat m = image_reader(image_path);
  CHECK_IMAGE(m);

  // step 2: extract feature by config and append to feature vector
  for (const auto &cfg : extract_config) {
    auto &feature_extractor = kFeatureExtractors.at(cfg[kFeatureExtractorKey]);
    auto &splitter = kSplitFuncs.at(cfg[kSplitFuncKey]);
    std::vector<float> partial_feature;
    if (0 !=
        feature_extractor(splitter(m), cfg[kConfigKey], &partial_feature)) {
      LOG_ERROR("failed to extract feature from " << image_path);
      break;
    }
    feature->insert(feature->end(), partial_feature.begin(),
                    partial_feature.end());
  }
  return 0;
}
