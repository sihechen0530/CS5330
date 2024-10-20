/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * header file for segmentation operation related functions
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <opencv2/opencv.hpp>

/**
 * Applies a threshold to the source image, converting it to a binary image.
 *
 * @param src The input source image (grayscale or color).
 * @param binary Pointer to the output binary image (thresholded result).
 * @return Status code (0 for success, or an error code).
 */
int threshold(const cv::Mat &src, cv::Mat *binary);

/**
 * Cleans the binary image by removing noise and unwanted small regions.
 *
 * @param binary The input binary image (thresholded result).
 * @param cleaned Pointer to the output cleaned binary image.
 * @return Status code (0 for success, or an error code).
 */
int clean(const cv::Mat &binary, cv::Mat *cleaned);

/**
 * Segments the cleaned image into distinct regions (connected components).
 *
 * @param cleaned The input cleaned binary image.
 * @param prev Pointer to the previous segmentation result (can be used for
 * tracking changes).
 * @param regions Pointer to the output segmentation result (labelled regions).
 * @return Status code (0 for success, or an error code).
 */
int segment(const cv::Mat &cleaned, cv::Mat *prev, cv::Mat *regions);

/**
 * Visualizes the segmentation result by color-coding or overlaying regions.
 *
 * @param segmentation The input segmentation result (labelled regions).
 * @param segvis Pointer to the output visualized segmentation image.
 * @return Status code (0 for success, or an error code).
 */
int segmentVisualize(const cv::Mat &segmentation, cv::Mat *segvis);

#endif
