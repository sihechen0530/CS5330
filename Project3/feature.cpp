/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * source file for feature related functions
 */

#include "feature.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_set>

constexpr int kNearestNeighborK = 3;

std::vector<float> normalize(const std::vector<float> &v) {
  float norm = std::sqrt(
      std::accumulate(v.begin(), v.end(), 0.0f, [](float ret, const auto &ele) {
        return ret + ele * ele;
      }));
  std::vector<float> normalizedVec(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    normalizedVec[i] = v[i] / norm;
  }
  return normalizedVec;
};

float cosine(const std::vector<float> &v1, const std::vector<float> &v2) {
  ASSERT_EQ_RET(v1.size(), v2.size(), -1);
  std::vector<float> normVec1 = normalize(v1);
  std::vector<float> normVec2 = normalize(v2);
  float dotProduct = 0.0;
  for (size_t i = 0; i < normVec1.size(); ++i) {
    dotProduct += normVec1[i] * normVec2[i];
  }
  return 1 - dotProduct;
}

int loadDatabase(const std::string &filename,
                 std::vector<ObjectFeature> *database) {
  std::ifstream file(filename);
  if (!file) {
    LOG_ERROR("Database file not found. Creating new one.");
    return -1;
  }
  std::string line;

  // Read the file line by line
  while (getline(file, line)) {
    ObjectFeature of;
    std::vector<float> lineData; // Vector to store data from a single line
    std::stringstream ss(line);  // String stream to parse the line
    ss >> of.category;
    float value;
    while (ss >> value) {
      of.feature.emplace_back(value);
    }
    database->emplace_back(of);
  }
  file.close();
  return 0;
}

int saveToDatabase(const std::vector<ObjectFeature> &database,
                   const std::string &dbFilename) {
  std::ofstream file(dbFilename, std::ios::out);
  for (const auto &feature : database) {
    LOG_INFO("save to file " << feature.category);
    file << feature.category << " ";
    for (const auto &feature_val : feature.feature) {
      file << feature_val << " ";
    }
    file << std::endl;
  }
  file.close();
  return 0;
}

bool matchFeature(const ObjectFeature &target,
                  const std::vector<ObjectFeature> &database,
                  std::string *matchedCategory, float *minCosineDist) {
  bool found = false;
  *minCosineDist = 1.0; // Maximum possible cosine distance
  for (const auto &dbFeature : database) {
    float dist = cosine(target.feature, dbFeature.feature);
    if (dist < *minCosineDist) {
      *minCosineDist = dist;
      *matchedCategory = dbFeature.category;
      found = true;
    }
  }
  return found;
}

bool matchFeatureKNN(const ObjectFeature &target,
                     const std::vector<ObjectFeature> &database,
                     std::string *matchedCategory, float *totalDistance) {
  if (database.empty()) {
    return false;
  }
  cv::Mat features;
  cv::Mat labels(database.size(), 1, CV_32S);
  std::unordered_map<std::string, int> categoryMap;
  std::vector<std::string> categories;
  for (size_t i = 0; i < database.size(); ++i) {
    const auto &feature = database.at(i).feature;
    const auto &category = database.at(i).category;
    cv::Mat row = cv::Mat(feature).reshape(1, 1);
    features.push_back(row);
    if (categoryMap.find(category) == categoryMap.end()) {
      categoryMap[category] = (int)categories.size();
      categories.emplace_back(category);
    }
    labels.at<int>(i, 0) = categoryMap[category];
  }

  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->train(features, cv::ml::ROW_SAMPLE, labels);

  cv::Mat newFeature = cv::Mat(target.feature).reshape(1, 1);

  cv::Mat result;
  cv::Mat neighborIndices;
  cv::Mat dists;
  knn->findNearest(newFeature, kNearestNeighborK, neighborIndices, result,
                   dists);

  *matchedCategory = categories[static_cast<int>(result.at<float>(0, 0))];

  for (int i = 0; i < dists.rows; ++i) {
    *totalDistance += dists.at<float>(i, 0);
  }
  return true;
}

int getRotatedRectFromCorners(const std::vector<cv::Point2f> &boxCorners,
                              cv::RotatedRect *obb) {
  // Calculate the center as the midpoint of the diagonal
  cv::Point2f center = (boxCorners[0] + boxCorners[2]) * 0.5f;

  // Calculate the width and height of the rotated rectangle
  float width = cv::norm(boxCorners[0] - boxCorners[1]);
  float height = cv::norm(boxCorners[1] - boxCorners[2]);

  // Calculate the angle of the rotated rectangle using the first edge (corner 0
  // to corner 1)
  float angle = std::atan2(boxCorners[1].y - boxCorners[0].y,
                           boxCorners[1].x - boxCorners[0].x) *
                180.0 / CV_PI;

  // Return the RotatedRect object
  *obb = cv::RotatedRect(center, cv::Size2f(width, height), angle);
  return 0;
}

int extractOneRegion(const int regionID, const cv::Mat &regions,
                     cv::RotatedRect *obb, ObjectFeature *feature) {
  cv::Mat roiImage = (regions == regionID);
  cv::Moments moments = cv::moments(roiImage, true);
  if (moments.m00 == 0) {
    LOG_ERROR("Empty or invalid region!");
    return -1;
  }
  float cx = moments.m10 / moments.m00;
  float cy = moments.m01 / moments.m00;
  float theta = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);
  float cosTheta = std::cos(theta);
  float sinTheta = std::sin(theta);

  int minX = roiImage.cols, minY = roiImage.rows;
  int maxX = 0, maxY = 0;

  for (int y = 0; y < roiImage.rows; ++y) {
    for (int x = 0; x < roiImage.cols; ++x) {
      if (roiImage.at<uint8_t>(y, x) > 0) { // Foreground pixel

        // Translate the point to the centroid
        float xTranslated = x - cx;
        float yTranslated = y - cy;

        // Rotate the point
        float xRotated = xTranslated * cosTheta + yTranslated * sinTheta;
        float yRotated = -xTranslated * sinTheta + yTranslated * cosTheta;

        // Update the extremal points in the rotated space
        if (xRotated < minX)
          minX = xRotated;
        if (xRotated > maxX)
          maxX = xRotated;
        if (yRotated < minY)
          minY = yRotated;
        if (yRotated > maxY)
          maxY = yRotated;
      }
    }
  }

  std::vector<cv::Point2f> boxCorners(4);
  boxCorners[0] = cv::Point2f(minX, minY);
  boxCorners[1] = cv::Point2f(maxX, minY);
  boxCorners[2] = cv::Point2f(maxX, maxY);
  boxCorners[3] = cv::Point2f(minX, maxY);

  for (int i = 0; i < 4; ++i) {
    float xOrig = boxCorners[i].x * cosTheta - boxCorners[i].y * sinTheta + cx;
    float yOrig = boxCorners[i].x * sinTheta + boxCorners[i].y * cosTheta + cy;
    boxCorners[i] = cv::Point2f(xOrig, yOrig);
  }

  getRotatedRectFromCorners(boxCorners, obb);

  float boundingBoxArea = obb->size.width * obb->size.height;
  float regionArea = moments.m00;
  float percentFilled = regionArea / boundingBoxArea;
  float aspectRatio = obb->size.height / obb->size.width;
  feature->feature.emplace_back(boundingBoxArea);
  feature->feature.emplace_back(regionArea);
  feature->feature.emplace_back(percentFilled);
  feature->feature.emplace_back(aspectRatio);
  return 0;
}

int extract(const cv::Mat &regions, std::vector<cv::RotatedRect> *obbs,
            std::vector<ObjectFeature> *features) {
  std::unordered_set<uint8_t> region_ids;
  for (int i = 0; i < regions.rows; ++i) {
    for (int j = 0; j < regions.cols; ++j) {
      uint8_t region_id = regions.at<uint8_t>(i, j);
      if (region_id != 0 && region_ids.find(region_id) == region_ids.end()) {
        obbs->emplace_back();
        features->emplace_back();
        extractOneRegion(region_id, regions, &obbs->back(), &features->back());
        region_ids.insert(region_id);
      }
    }
  }
  return 0;
}

void computeOrientedBoundingBox(const cv::Mat &binaryImage, cv::Mat &output) {
  // Find contours to get the shape of the region
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    std::cerr << "No contours found!" << std::endl;
    return;
  }

  // Use the first contour (assuming there's only one region)
  std::vector<cv::Point> contour = contours[0];

  // Calculate the oriented bounding box using minAreaRect
  cv::RotatedRect obb = cv::minAreaRect(contour);

  // Draw the oriented bounding box
  cv::Point2f vertices[4];
  obb.points(vertices);
  for (int i = 0; i < 4; i++) {
    cv::line(output, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0),
             2);
  }

  // Display results
  cv::imshow("Oriented Bounding Box", output);
  cv::waitKey(0);
}
