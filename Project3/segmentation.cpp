/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * source file for segmentation operation related functions
 */

#include "segmentation.h"
#include "utils.h"
#include <unordered_map>
#include <vector>

constexpr int kGaussianKernelSize = 5;
constexpr int kSampleSize = 4;
constexpr int kClusterCount = 2;
constexpr int kMaxAttempts = 3;
constexpr int kMaxRegionCount = 10;
constexpr int kMinRegionPixelCount = 50000;

int threshold(const cv::Mat &src, cv::Mat *binary) {
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred,
                   cv::Size(kGaussianKernelSize, kGaussianKernelSize), 0);
  int sample_rows = src.rows / kSampleSize;
  int sample_cols = src.cols / kSampleSize;
  cv::Mat samples(sample_rows * sample_cols, 1, CV_32F);
  for (int i = 0; i < sample_rows; ++i) {
    for (int j = 0; j < sample_cols; ++j) {
      samples.at<float>(i * sample_cols + j, 0) =
          src.at<uint8_t>(i * kSampleSize, j * kSampleSize);
    }
  }
  cv::Mat labels;
  cv::Mat centers;
  cv::kmeans(samples, kClusterCount, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              10, 1.0),
             kMaxAttempts, cv::KMEANS_PP_CENTERS, centers);
  float thresholdValue = static_cast<int>(
      (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2.0);
  cv::threshold(gray, *binary, thresholdValue, 255, cv::THRESH_BINARY);
  return 0;
}

int close(const cv::Mat &src, cv::Mat *dst) {
  int morph_size = 3;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_CROSS, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
      cv::Point(morph_size, morph_size));
  cv::morphologyEx(src, *dst, cv::MORPH_CLOSE, kernel);
  return 0;
}

int open(const cv::Mat &src, cv::Mat *dst) {
  int morph_size = 3;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_CROSS, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
      cv::Point(morph_size, morph_size));
  cv::morphologyEx(src, *dst, cv::MORPH_OPEN, kernel);
  return 0;
}

int clean(const cv::Mat &binary, cv::Mat *cleaned) {
  // fill holes and remove noise
  // closing
  close(binary, cleaned);
  return 0;
}

int twoPassSegment(const cv::Mat &binary, cv::Mat *labeled) {
  int label = 1;
  std::unordered_map<int, int> union_set;
  for (int i = 0; i < binary.rows; i++) {
    for (int j = 0; j < binary.cols; j++) {
      if (binary.at<uchar>(i, j) == 0) { // Foreground pixel
        int left = (j > 0) ? labeled->at<int>(i, j - 1) : 0;
        int top = (i > 0) ? labeled->at<int>(i - 1, j) : 0;

        if (left == 0 && top == 0) {
          labeled->at<int>(i, j) = label;
          ++label;
        } else if (left != 0 && top == 0) {
          labeled->at<int>(i, j) = left;
        } else if (left == 0 && top != 0) {
          labeled->at<int>(i, j) = top;
        } else {
          labeled->at<int>(i, j) = std::min(left, top);
          if (left != top) {
            union_set[std::max(left, top)] = std::min(left, top);
          }
        }
      }
    }
  }

  for (int i = 0; i < labeled->rows; i++) {
    for (int j = 0; j < labeled->cols; j++) {
      if (labeled->at<int>(i, j) != 0) {
        int currentLabel = labeled->at<int>(i, j);
        while (union_set.find(currentLabel) != union_set.end()) {
          currentLabel = union_set[currentLabel];
        }
        labeled->at<int>(i, j) = currentLabel;
      }
    }
  }
  return 0;
}

int opencvFindContours(const int regionID, const cv::Mat &binary,
                       cv::Moments *moments) {
  // Find contours to isolate each region
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  // Get the specific contour for the given region ID
  std::vector<cv::Point> regionContour = contours[regionID];

  // Compute moments of the region
  *moments = cv::moments(regionContour);
  return 0;
}

int keepKRegions(const int max_region_count, cv::Mat *regions) {
  std::unordered_map<int, int> frequency;
  for (int i = 0; i < regions->rows; i++) {
    for (int j = 0; j < regions->cols; j++) {
      const int region_id = regions->at<int>(i, j);
      if (region_id != 0) {
        ++frequency[region_id];
      }
    }
  }
  std::vector<std::pair<int, int>> index_count;
  for (const auto &[region_id, count] : frequency) {
    if (count <= kMinRegionPixelCount) {
      continue;
    }
    index_count.emplace_back(region_id, count);
  }
  int index_count_len = index_count.size();
  if (index_count_len > max_region_count) {
    std::nth_element(index_count.begin(),
                     index_count.begin() + max_region_count, index_count.end(),
                     [](const auto &lhs, const auto &rhs) {
                       return lhs.second > rhs.second;
                     });
  }
  std::unordered_map<int, int> region_id_map;
  int new_region_id = 1;
  for (int i = 0; i < std::min(index_count_len, max_region_count); ++i) {
    region_id_map[index_count[i].first] = new_region_id;
    ++new_region_id;
  }
  for (int i = 0; i < regions->rows; ++i) {
    for (int j = 0; j < regions->cols; ++j) {
      if (region_id_map.find(regions->at<int>(i, j)) == region_id_map.end()) {
        regions->at<int>(i, j) = 0;
      } else {
        regions->at<int>(i, j) = region_id_map[regions->at<int>(i, j)];
      }
    }
  }
  return 0;
}

int grassFire(const cv::Mat &regions, int region_id, int *y, int *x) {
  cv::Mat distance = cv::Mat::zeros(regions.size(), CV_32S);
  for (int i = 0; i < regions.rows; ++i) {
    for (int j = 0; j < regions.cols; ++j) {
      if (regions.at<uint8_t>(i, j) == region_id) {
        int up = 0, left = 0;
        if (i - 1 >= 0 && regions.at<uint8_t>(i - 1, j) == region_id) {
          up = distance.at<int>(i - 1, j);
        }
        if (j - 1 >= 0 && regions.at<uint8_t>(i, j - 1) == region_id) {
          left = distance.at<int>(i, j - 1);
        }
        distance.at<int>(i, j) = std::min(up, left) + 1;
      }
    }
  }
  for (int i = regions.rows - 1; i >= 0; --i) {
    for (int j = regions.cols - 1; j >= 0; --j) {
      if (regions.at<uint8_t>(i, j) == region_id) {
        int down = 0, right = 0;
        if (i + 1 < regions.rows &&
            regions.at<uint8_t>(i + 1, j) == region_id) {
          down = distance.at<int>(i + 1, j);
        }
        if (j + 1 < regions.cols &&
            regions.at<uint8_t>(i, j + 1) == region_id) {
          right = distance.at<int>(i, j + 1);
        }
        distance.at<int>(i, j) = std::min(down, right) + 1;
      }
    }
  }
  int max_dist = 0;
  for (int i = 0; i < distance.rows; ++i) {
    for (int j = 0; j < distance.cols; ++j) {
      if (distance.at<int>(i, j) > max_dist) {
        max_dist = distance.at<int>(i, j);
        *y = i;
        *x = j;
      }
    }
  }
  return 0;
}

int regionIdMatch(const cv::Mat &prev, cv::Mat *regions) {
  // regions and prev have 0-255 regions
  // compute centroids of each regions in regions and use the region id of prev
  std::unordered_map<int, int> region_id_map;
  for (int r = 1; r < kMaxRegionCount; ++r) {
    bool has_processed = false;
    for (int i = 0; i < regions->rows; ++i) {
      if (has_processed) {
        break;
      }
      for (int j = 0; j < regions->cols; ++j) {
        if (regions->at<uint8_t>(i, j) == r) {
          cv::Mat roiImage = (*regions == r);
          cv::Moments moments = cv::moments(roiImage, true);
          float cx = 0, cy = 0;
          if (moments.m00 != 0) {
            cx = moments.m10 / moments.m00;
            cy = moments.m01 / moments.m00;
          }
          region_id_map[r] = prev.at<uint8_t>((int)cy, int(cx));
          has_processed = true;
          break;
        }
      }
    }
  }
  for (int i = 0; i < regions->rows; ++i) {
    for (int j = 0; j < regions->cols; ++j) {
      const uint8_t orig_id = regions->at<uint8_t>(i, j);
      if (region_id_map.find(orig_id) != region_id_map.end()) {
        regions->at<uint8_t>(i, j) = region_id_map[orig_id];
      }
    }
  }

  return 0;
}

int segment(const cv::Mat &cleaned, cv::Mat *prev, cv::Mat *regions) {
  // run connected component analysis
  // use previous frame id to avoid color flickering
  *regions = cv::Mat::zeros(cleaned.size(), CV_32S);
  twoPassSegment(cleaned, regions);
  // keep k regions
  keepKRegions(kMaxRegionCount, regions);
  regions->convertTo(*regions, CV_8UC1);
  // region id matching
  // if (!prev->empty()) {
  //   regionIdMatch(*prev, regions);
  // }
  // *prev = regions->clone();
  return 0;
}

int segmentVisualize(const cv::Mat &segmentation, cv::Mat *segvis) {
  // segmentation: region id from 0-kMaxRegionCount
  segmentation.convertTo(*segvis, CV_8UC1, 255 / kMaxRegionCount);
  cv::Mat mask = (*segvis == 0);
  segvis->setTo(255, mask);
  return 0;
}
