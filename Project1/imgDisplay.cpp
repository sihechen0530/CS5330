/*
  Sihe Chen (002085773)
  Fall 2024
  CS 5330 Project 1
  main function for image display
*/
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr char QUIT_KEY = 'q';

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "usage: imgDisplay <Image_Path>" << std::endl;
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[1], cv::IMREAD_COLOR);

  if (!image.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);

  std::cout << "Press q to quit" << std::endl;
  while (true) {
    int key = cv::waitKey(0);

    if (key == QUIT_KEY) {
      break;
    }
  }
  return 0;
}
