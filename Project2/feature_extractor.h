/*
  Sihe Chen (002085773)
  Fall 2024
  CS 5330 Project 2
  header file for multiple feature extractors
*/
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// The functions below are for getting a region of the image.

/**
 * Get the whole image out of the image.
 *
 * This function takes an input image `src` and return the whole src matrix.
 *
 * @param src The original input image.
 * @return The whole input image.
 */
cv::Mat getWholeImage(const cv::Mat &src);

/**
 * Get the upper half of the image.
 *
 * This function takes an input image `src` and return the upper half of the src
 * image as an output.
 *
 * @param src The original input image.
 * @return The upper half of the input image.
 */
cv::Mat getUpperHalf(const cv::Mat &src);

/**
 * Get the lower half of the image.
 *
 * This function takes an input image `src` and return the lower half of the src
 * image as an output.
 *
 * @param src The original input image.
 * @return The lower half of the input image.
 */
cv::Mat getLowerHalf(const cv::Mat &src);

/**
 * Get the left half of the image.
 *
 * This function takes an input image `src` and return the left half of the src
 * image as an output.
 *
 * @param src The original input image.
 * @return The left half of the input image.
 */
cv::Mat getLeftHalf(const cv::Mat &src);

/**
 * Get the right half of the image.
 *
 * This function takes an input image `src` and return the right half of the src
 * image as an output.
 *
 * @param src The original input image.
 * @return The right half of the input image.
 */
cv::Mat getRightHalf(const cv::Mat &src);

/**
 * Get the upper left quarter of the image.
 *
 * This function takes an input image `src` and return the upper left quarter of
 * the src image as an output.
 *
 * @param src The original input image.
 * @return The upper left quarter of the input image.
 */
cv::Mat getUpperLeft(const cv::Mat &src);

/**
 * Get the upper right quarter of the image.
 *
 * This function takes an input image `src` and return the upper left quarter of
 * the src image as an output.
 *
 * @param src The original input image.
 * @return The upper left quarter of the input image.
 */
cv::Mat getUpperRight(const cv::Mat &src);

/**
 * Get the lower left quarter of the image.
 *
 * This function takes an input image `src` and return the lower left quarter of
 * the src image as an output.
 *
 * @param src The original input image.
 * @return The lower left quarter of the input image.
 */
cv::Mat getLowerLeft(const cv::Mat &src);

/**
 * Get the lower right quarter of the image.
 *
 * This function takes an input image `src` and return the lower right quarter
 * of the src image as an output.
 *
 * @param src The original input image.
 * @return The lower right quarter of the input image.
 */
cv::Mat getLowerRight(const cv::Mat &src);

/**
 * The external entrance of all extraction methods. Loads an image from path,
 * apply multiple extraction methods, append the results and return as a
 * vector float.
 *
 * This function takes an input image path, extraction configuration in Json
 * format and return the extracted feature.
 *
 * @param image_path the file path of the image
 * @param extract_config the configuration of extraction in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int extract(const std::string &image_path, const nlohmann::json &extract_config,
            std::vector<float> *feature);

/**
 * Extract a region of image as feature vector. Simply concatenate all the rgb
 * values together.
 *
 * This function takes an input image path, extraction configuration in Json
 * format and return the extracted feature.
 *
 * @param image_path the file path of the image
 * @param extract_config the configuration of extraction in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int roi(const cv::Mat &m, const nlohmann::json &config,
        std::vector<float> *feature);

/**
 * Compute rg chromaticity histogram as feature vector.
 *
 * This function takes an input image matrix, bin_size as config in Json
 * and return the computed histogram.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int rgHistogram(const cv::Mat &m, const nlohmann::json &config,
                std::vector<float> *feature);

/**
 * Compute rgb value histogram as feature vector.
 *
 * This function takes an input image matrix, bin_size as config in Json
 * and return the computed histogram.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int rgbHistogram(const cv::Mat &m, const nlohmann::json &config,
                 std::vector<float> *feature);

/**
 * Compute sobel magnitude and angle histogram as feature vector.
 *
 * This function takes an input image matrix, bin_size as config in Json
 * and return the computed histogram.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int sobelHistogram(const cv::Mat &m, const nlohmann::json &config,
                   std::vector<float> *feature);

/**
 * Apply laws filter and compute histogram as feature vector.
 *
 * This function takes an input image matrix, bin_size as config in Json
 * and return the computed histogram.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int lawsFilterHistogram(const cv::Mat &m, const nlohmann::json &config,
                        std::vector<float> *feature);

/**
 * Apply gabor filter and compute histogram as feature vector.
 *
 * This function takes an input image matrix, bin_size / orientation / frequency
 * as config in Json and return the computed histogram.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int gaborFilterHistogram(const cv::Mat &m, const nlohmann::json &config,
                         std::vector<float> *feature);

/**
 * Apply discrete fourier transformation and resize the result by configuration.
 *
 * This function takes an input image matrix, resize size as config in Json
 * and return the FT result.
 *
 * @param m the image matrix
 * @param config the configuration in Json format
 * @param feature the pointer to the output feature vector
 * @return Whether the function runs successfully(0) or not(otherwise).
 */
int fourierTransform(const cv::Mat &m, const nlohmann::json &config,
                     std::vector<float> *feature);

#endif