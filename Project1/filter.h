/*
  Sihe Chen (002085773)
  Fall 2024
  CS 5330 Project 1
  header file for multiple filters
*/
#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>

/**
 * Converts an input image to grayscale.
 *
 * This function takes an input image `src` and converts it to grayscale using
 * cv::cvtColor. The resulting grayscale image is stored in the `res` matrix.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC1
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int convertGreyScale(const cv::Mat &src, cv::Mat &dst);

/**
 * Converts an input image to grayscale.
 *
 * This function takes an input image `orig` and converts it to grayscale using
 * max(R, G, B). The resulting grayscale image is stored in the `res` matrix.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC1
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int convertGreyScale2(const cv::Mat &src, cv::Mat &dst);

/**
 * Applies a sepia tone effect to an input image.
 *
 * This function takes an input image `src` and applies a sepia tone effect to
 * it. The calculation is 0.272*R + 0.534*G + 0.131*B. The resulting image with
 * the sepia tone effect is stored in the `dst` matrix.
 *
 * @param src The original input image to which the sepia tone effect will be
 * applied. This parameter is passed as a constant reference to a `cv::Mat`
 * object.
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int convertSepiaTone(const cv::Mat &src, cv::Mat &dst);

/**
 * Naive implementation of 5x5 Gaussian blur filter.
 *
 * This function takes an input image `src` and applies a 5x5 blur filter to it.
 * The resulting blurred image is stored in the `dst` matrix.
 *
 * The approximate time of one execution is 0.25s.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int blur5x5_1(const cv::Mat &src, cv::Mat &dst);

/**
 * Accelerated implementation of 5x5 Gaussian blur filter.
 *
 * This function takes an input image `src` and applies a 5x5 blur filter to it.
 * The resulting blurred image is stored in the `dst` matrix.
 *
 * The approximate time of one execution is 0.04s.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int blur5x5_2(const cv::Mat &src, cv::Mat &dst);

/**
 * Applies the horizontal Sobel operator with a 3x3 kernel to an input image.
 *
 * This function takes an input image `src` and applies the horizontal Sobel
 * operator with a 3x3 kernel to it. The resulting image with the horizontal
 * Sobel operator applied is stored in the `dst` matrix.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_16SC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int sobelX3x3(const cv::Mat &src, cv::Mat &dst);

/**
 * Applies the vertical Sobel operator with a 3x3 kernel to an input image.
 *
 * This function takes an input image `src` and applies the vertical Sobel
 * operator with a 3x3 kernel to it. The resulting image with the vertical
 * Sobel operator applied is stored in the `dst` matrix. It cannot be directly
 * shown in `cv::imshow` before going through `sobelVisualize` function.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_16SC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int sobelY3x3(const cv::Mat &src, cv::Mat &dst);

/**
 * Visualize the Sobel filtered result.
 *
 * This function takes an input image `src` which is the result of Sobel X/Y
 * filter. The function convert the scale of `src`, which is [-255, 255], to a
 * uint8_t type [0, 255]. The result is then stored in the `dst` matrix.
 * *
 * @param src The original input image. CV_16SC3
 * @param dst The resulting image. CV_8UC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int sobelVisualize(const cv::Mat &src, cv::Mat &dst);

/**
 * Compute the magnitude of Sobel X and Sobel Y filtered result.
 *
 * This function takes two input images, sx (result of Sobel X filter) and sy
 * (result of Sobel Y filter) and computes the magnitude. The magnitude is
 * computed by `sqrt(sx * sx, sy * sy)`. The result is stored in the `dst`
 * matrix.
 *
 * @param sx The result of Sobel X filter. CV_16SC3
 * @param sy The result of Sobel Y filter. CV_16SC3
 * @param dst The resulting image. CV_8UC3
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst);

/**
 * Applies blur and quantization to an input image.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC3
 * @param levels The number of quantization levels as an integer. Default value
 * is 10.
 * @return An integer value indicating the success or failure of the conversion.
 * A return value of 0 indicates success, while a non-zero value indicates
 * failure.
 */
int blurQuantize(const cv::Mat &src, cv::Mat &dst, const int levels = 10);

/**
 * Applies frosted glass distortion effect to an input image.
 *
 * @param input The original input image. CV_8UC3
 * @param distortionStrength The strength of the distortion as an integer.
 * @param output The resulting image. CV_8UC3
 * @return An integer value representing the result of the operation.
 */
int glassDistort(const cv::Mat &input, const int distortionStrength,
                 cv::Mat &output);

/**
 * Increases the brightness of an input image by adding a specified step to the
 * V value of HSV color space.
 *
 * @param src The original input image. CV_8UC3
 * @param step The step value to increase the brightness as an integer.
 * @param dst The resulting image. CV_8UC3
 * @return An integer value representing the result of the operation.
 */
int brighten(const cv::Mat &src, const int step, cv::Mat &dst);

/**
 * Improve the facial quality by smoothening the skin and lighten up the facial
 * region.
 *
 * @param src The original input image. CV_8UC3
 * @param faces Detected facial regions stored in vector of cv::Rect.
 * @param dst The resulting image. CV_8UC3
 * @return An integer value representing the result of the operation.
 */
int improveFace(const cv::Mat &src, const std::vector<cv::Rect> &faces,
                cv::Mat &dst);

/**
 * Applies a time delay effect to a time range of input images.
 * The effect is achieved by blending the current frame with the previous frame.
 *
 * @param src The original input image. CV_8UC3
 * @param dst The resulting image. CV_8UC3
 * @param count The number of frames to delay.
 * @param alpha The weight of a single frame when blended into the resulting
 * image.
 * @return An integer value representing the result of the operation.
 */
class TimeDelay {
public:
  TimeDelay(int count, float alpha) : count_(count), alpha_(alpha) {}
  ~TimeDelay() = default;
  int timeDelay(const cv::Mat &src, cv::Mat &dst);

private:
  int count_;
  float alpha_;
  std::deque<cv::Mat> frames_;
};

#endif // FILTER_HPP