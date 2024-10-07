#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat getWholeImage(const cv::Mat &src);

cv::Mat getUpperHalf(const cv::Mat &src);

cv::Mat getLowerHalf(const cv::Mat &src);

cv::Mat getLeftHalf(const cv::Mat &src);

cv::Mat getRightHalf(const cv::Mat &src);

cv::Mat getUpperLeft(const cv::Mat &src);

cv::Mat getUpperRight(const cv::Mat &src);

cv::Mat getLowerLeft(const cv::Mat &src);

cv::Mat getLowerRight(const cv::Mat &src);

int extract(const std::string &image_path, const nlohmann::json &extract_config,
            std::vector<float> *feature);

int roi(const cv::Mat &m, const nlohmann::json &config,
        std::vector<float> *feature);

int rgHistogram(const cv::Mat &m, const nlohmann::json &config,
                std::vector<float> *feature);

int rgbHistogram(const cv::Mat &m, const nlohmann::json &config,
                 std::vector<float> *feature);

int sobelHistogram(const cv::Mat &m, const nlohmann::json &config,
                   std::vector<float> *feature);

#endif