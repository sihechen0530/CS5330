/**
 * Sihe Chen (002085773)
 * Fall 2024
 * CS 5330 Project 3
 * main function
 */

#include "feature.h"
#include "segmentation.h"
#include "utils.h"
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

const std::string kThreshold = "threshold";
const std::string kClean = "clean";
const std::string kSegmentation = "segmentation";
const std::string kSegmentationVis = "segmentationVisualization";
const std::string kBBox = "boundingbox";
const std::string kMatching = "matching";

constexpr char kQuitKey = 'q';
constexpr char kSpaceKey = ' ';
const std::string kOrigTitle = "original";

// the object bounding boxes and features of the latest frame
std::vector<cv::RotatedRect> obbs;
std::vector<ObjectFeature> features;
std::vector<ObjectFeature> database;
cv::Mat frame;
bool paused = false;

// mouse click event callback: get user input and save to database
void onMouse(int event, int x, int y, int, void *) {
  ASSERT_EQ_RET(obbs.size(), features.size(), );
  if (event == cv::EVENT_LBUTTONDOWN) {
    for (size_t i = 0; i < obbs.size(); i++) {
      if (isPointInRotatedRect(cv::Point(x, y), obbs[i])) {
        LOG_INFO("Box " << i + 1 << " clicked!");
        GET_INPUT_WITH_PROMPT("Enter new category for this object: ",
                              features[i].category);
        database.push_back(features[i]);
        break;
      }
    }
  }
}

// match with database. draw bounding boxes.
int matching(const cv::Mat &src, const cv::Mat &segmentation,
             const std::vector<ObjectFeature> &database, cv::Mat *bbox) {
  obbs.clear();
  features.clear();
  extract(segmentation, &obbs, &features);
  ASSERT_EQ_RET(obbs.size(), features.size(), -1);
  src.copyTo(*bbox);
  for (size_t idx = 0; idx < features.size(); ++idx) {
    const auto &feature = features.at(idx);
    const auto &obb = obbs.at(idx);
    cv::Point2f vertices[4];
    obb.points(vertices);
    std::string matchedCategory;
    float minCosineDist = 0.0;
    for (int i = 0; i < 4; i++) {
      cv::line(*bbox, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0),
               2);
    }
    cv::circle(*bbox, obb.center, 5, cv::Scalar(255, 0, 0), -1);
    // use matchFeature() with same argument for cosine distance best match
    if (matchFeatureKNN(feature, database, &matchedCategory, &minCosineDist)) {
      // Match found, print the category and cosine distance
      LOG_INFO("Matched Category: " << matchedCategory
                                    << " Cosine Distance: " << minCosineDist);
      cv::putText(*bbox,
                  matchedCategory + " (" + std::to_string(minCosineDist) + ")",
                  cv::Point(std::clamp((int)vertices[0].x, 0, bbox->cols),
                            std::clamp((int)vertices[0].y, 0, bbox->rows)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    } else {
      cv::putText(*bbox, std::to_string(idx), vertices[0],
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
  }
  return 0;
}

// steps to process a frame: threshold, clean, segmentation, matching
int process_one_frame(const std::vector<ObjectFeature> &database,
                      const cv::Mat &src, cv::Mat *prev_segmentation) {
  std::unordered_map<std::string, cv::Mat> processed;
  cv::Mat thresholded;
  threshold(src, &thresholded);
  // showImage(kThreshold, thresholded);
  cv::Mat cleaned;
  clean(thresholded, &cleaned);
  showImage(kClean, cleaned);
  cv::Mat segmentation;
  segment(cleaned, prev_segmentation, &segmentation);
  // showImage(kSegmentation, segmentation);
  cv::Mat segvis;
  segmentVisualize(segmentation, &segvis);
  showImage(kSegmentationVis, segvis);
  // extract and match; if not matched, query from user prompt
  cv::Mat match;
  matching(src, segmentation, database, &match);
  showImage(kMatching, match);
  return 0;
}

int main(int argc, char *argv[]) {
  cv::VideoCapture camera(0);
  std::string dbFilename = "object_database.txt";
  loadDatabase(dbFilename, &database);

  cv::namedWindow(kMatching);
  cv::setMouseCallback(kMatching, onMouse);
  // open the video device
  if (!camera.isOpened()) {
    std::cout << "Unable to open video device" << std::endl;
    return (-1);
  }

  camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

  // get some properties of the image
  cv::Size refS((int)camera.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)camera.get(cv::CAP_PROP_FRAME_HEIGHT));
  std::cout << "Expected size: " << refS.width << " " << refS.height
            << std::endl;

  cv::Mat prev_segmentation;
  for (;;) {
    if (!paused) {
      camera >> frame;
      if (frame.empty()) {
        std::cout << "frame is empty" << std::endl;
        break;
      }
      showImage(kOrigTitle, frame);
      process_one_frame(database, frame, &prev_segmentation);
    }
    // see if there is a waiting keystroke
    char new_key = cv::waitKey(10);
    if (new_key >= 0 && kQuitKey == new_key) {
      break;
    }
    if (kSpaceKey == new_key)
      paused = !paused;
  }

  saveToDatabase(database, dbFilename);
  camera.release();
  cv::destroyAllWindows();
  return 0;
}
