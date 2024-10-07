#include "utils.h"


cv::Mat image_reader(const std::string &image_path) {
  return cv::imread(image_path, cv::IMREAD_COLOR);
}

int showMatches(const std::string &directory,
                const std::vector<std::string> &matches) {
  for (auto &match : matches) {
    auto realpath = directory + "/" + match;
    cv::Mat image = image_reader(realpath);
    cv::namedWindow(match, cv::WINDOW_AUTOSIZE);
    cv::imshow(match, image);
    LOG_INFO("showing" << realpath);
  }
  return 0;
}
