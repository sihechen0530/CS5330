#ifndef CALIBRATION_H
#define CALIBRATION_H
#include "utils.h"
#include <opencv2/opencv.hpp>

/**
 * extract chessboard corners and 3D points and store them in the corner set and
 * point set
 * @param src source image
 * @param dst destination image
 * @param point_set 3D point set
 * @param corner_set chessboard corner 2D point set
 * @return 0 if successful; else unsuccessful
 */
int getChessBoardCorners(const cv::Mat &src, const int corner_row,
                         const int corner_col, cv::Mat *dst,
                         std::vector<cv::Vec3f> *point_set,
                         std::vector<cv::Point2f> *corner_set);

/**
 * save the point list and corner list into file
 * @param point_list 3D points from multiple captures
 * @param corner_list 2D points from multiple captures
 * @param file_path the file path to save these points
 * @return 0 if successful; else unsuccessful
 */
int saveCornersToFile(const std::vector<std::vector<cv::Vec3f>> &point_list,
                      const std::vector<std::vector<cv::Point2f>> &corner_list,
                      const char *file_path);

/**
 * load point list and corner list from file
 * @param file_path file path to load from
 * @param point_list 3D points from multiple captures
 * @param corner_list 2D points from multiple captures
 * @return 0 if successful; else unsuccessful
 */
int loadCornersFromFile(const char *file_path,
                        std::vector<std::vector<cv::Vec3f>> *point_list,
                        std::vector<std::vector<cv::Point2f>> *corner_list);

/**
 * do calibration with points from at least kMinCalibPointCount captures
 * @param point_list 3D points from multiple captures
 * @param corner_list 2D points from multiple captures
 * @param image_size the size of input image
 * @param cameraMatrix intrinsic matrix
 * @return 0 if successful; else unsuccessful
 */
int calibrate(const std::vector<std::vector<cv::Vec3f>> &point_list,
              const std::vector<std::vector<cv::Point2f>> &corner_list,
              const cv::Size image_size, cv::Mat *cameraMatrix);

/**
 * get camera pose from point set and cornet set
 * @param point_set 3D points from one capture
 * @param corner_set 2D points from one capture
 * @param cameraMatrix intrinsic matrix of camera
 * @param rvec rotation vector
 * @param tvec translation vector
 * @return 0 if successful; else unsuccessful
 */
int getPose(const std::vector<cv::Vec3f> &point_set,
            const std::vector<cv::Point2f> &corner_set,
            const cv::Mat &cameraMatrix, cv::Mat *rvec, cv::Mat *tvec);

/**
 *  project point on image with intrinsic matrix
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param cameraMatrix intrinsic matrix
 * @param image output image
 * @return 0 if successful; else unsuccessful
 */
int projectPoints(const cv::Mat &rvec, const cv::Mat &tvec,
                  const cv::Mat &cameraMatrix, cv::Mat *image);
#endif
