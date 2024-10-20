/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * header file for feature related functions
 */

#ifndef FEATURE_H
#define FEATURE_H
#include <opencv2/opencv.hpp>
#include <vector>

// the feature values of an object of a specific category
typedef struct {
  std::string category;
  std::vector<float> feature;
} ObjectFeature;

/**
 * Extracts oriented bounding boxes (OBBs) and features from segmented regions.
 *
 * @param regions The input segmentation result (labelled regions).
 * @param obbs Pointer to the output vector of oriented bounding boxes
 * (RotatedRects).
 * @param features Pointer to the output vector of extracted object features.
 * @return Status code (0 for success, or an error code).
 */
int extract(const cv::Mat &regions, std::vector<cv::RotatedRect> *obbs,
            std::vector<ObjectFeature> *features);

/**
 * Matches a target object's feature to a feature database using cosine
 * similarity.
 *
 * @param target The input object feature to be matched.
 * @param database The database of object features to match against.
 * @param matchedCategory Pointer to the output matched category as a string.
 * @param minCosineDist Pointer to the output minimum cosine distance between
 * the target and matched feature.
 * @return Boolean indicating whether a match was found (true for match, false
 * for no match).
 */
bool matchFeature(const ObjectFeature &target,
                  const std::vector<ObjectFeature> &database,
                  std::string *matchedCategory, float *minCosineDist);

/**
 * Matches a target object's feature to a feature database using K-Nearest
 * Neighbors (KNN) with Euclidean distance.
 *
 * @param target The input object feature to be matched.
 * @param database The database of object features to match against.
 * @param matchedCategory Pointer to the output matched category as a string.
 * @param totalDistance Pointer to the output total distance between the target
 * and the nearest neighbors.
 * @return Boolean indicating whether a match was found (true for match, false
 * for no match).
 */
bool matchFeatureKNN(const ObjectFeature &target,
                     const std::vector<ObjectFeature> &database,
                     std::string *matchedCategory, float *totalDistance);

/**
 * Loads object feature data from a file into a feature database.
 *
 * @param filename The name of the file from which to load the database.
 * @param database Pointer to the output vector where the database of features
 * will be stored.
 * @return Status code (0 for success, or an error code).
 */
int loadDatabase(const std::string &filename,
                 std::vector<ObjectFeature> *database);

/**
 * Saves object features to a file, appending to the existing database.
 *
 * @param feature The vector of features to be saved.
 * @param filename The name of the file to which the database will be saved.
 * @return Status code (0 for success, or an error code).
 */
int saveToDatabase(const std::vector<ObjectFeature> &feature,
                   const std::string &filename);
#endif
