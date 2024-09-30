/*
  Sihe Chen (002085773)
  9/23/2024
  Fall 2024
  main function for video display
*/
#include "faceDetect.h"
#include "filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr char kQuitKey = 'q';
constexpr char kGreyKey = 'g';
constexpr char kGrey2Key = 'h';
constexpr char kSepiaKey = 's';
constexpr char kColorKey = 'c';
constexpr char kBlur = 'b';
constexpr char kSobelX = 'x';
constexpr char kSobelY = 'y';
constexpr char kSobelMagnitude = 'm';
constexpr char kBlurQuantize = 'l';
constexpr char kFaceDetection = 'f';
constexpr char kGlassDistort = 'd';
constexpr char kTimeDelay = 't';
constexpr char kImproveFace = 'i';
constexpr char kSaveImage = 'p';
constexpr char kOrigTitle[] = "Original";
constexpr char kFilterTitle[] = "Filtered";
constexpr char kDefaultSavePath[] = "./img.jpg";

constexpr int kGlassDistortStrength = 20;
constexpr int kTimeDelayFrameCount = 20;
constexpr float kTimeDelayAlpha = 0.2f;

int main(int argc, char *argv[]) {
  const char *img_save_path;
  if (argc > 1) {
    img_save_path = argv[1];
  } else {
    img_save_path = kDefaultSavePath;
  }
  cv::VideoCapture camera(0);

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

  cv::Mat show;
  cv::Mat sobel_x_result, sobel_y_result;
  cv::Mat face_detect_grey;
  std::vector<cv::Rect> faces;
  TimeDelay time_delay(kTimeDelayFrameCount, kTimeDelayAlpha);

  char last_key = kColorKey;

  for (;;) {
    // std::cout << "##DEBUG last key is: " << last_key << std::endl;
    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) {
      std::cout << "frame is empty" << std::endl;
      break;
    }
    cv::imshow(kOrigTitle, frame);
    switch (last_key) {
    case kGreyKey:
      // std::cout << "Displaying Grey Image!" << std::endl;
      convertGreyScale(frame, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kGrey2Key:
      // std::cout << "Displaying Alternative Grey Image!" << std::endl;
      convertGreyScale2(frame, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kSepiaKey:
      // std::cout << "Displaying Sepia Tone Image!" << std::endl;
      convertSepiaTone(frame, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kBlur:
      // std::cout << "Displaying Blur 5x5 Image!" << std::endl;
      blur5x5_2(frame, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kSobelX:
      // std::cout << "Displaying Sobel X 3x3 Image!" << std::endl;
      sobelX3x3(frame, sobel_x_result);
      sobelVisualize(sobel_x_result, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kSobelY:
      // std::cout << "Displaying Sobel Y 3x3 Image!" << std::endl;
      sobelY3x3(frame, sobel_y_result);
      sobelVisualize(sobel_y_result, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kSobelMagnitude:
      // std::cout << "Displaying Sobel Magnitude Image!" << std::endl;
      sobelX3x3(frame, sobel_x_result);
      sobelY3x3(frame, sobel_y_result);
      magnitude(sobel_x_result, sobel_y_result, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kBlurQuantize:
      // std::cout << "Displaying Blur Quantize Image!" << std::endl;
      blurQuantize(frame, show, 3);
      cv::imshow(kFilterTitle, show);
      break;
    case kFaceDetection:
      // std::cout << "Displaying Face Detection Image!" << std::endl;
      cv::cvtColor(frame, face_detect_grey, cv::COLOR_BGR2GRAY, 0);
      detectFaces(face_detect_grey, faces);
      frame.copyTo(show);
      drawBoxes(show, faces);
      cv::imshow(kFilterTitle, show);
      break;
    case kGlassDistort:
      // std::cout << "Displaying Glass Distorted Image!" << std::endl;
      glassDistort(frame, kGlassDistortStrength, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kTimeDelay:
      // std::cout << "Displaying Time Delay Image!" << std::endl;
      time_delay.timeDelay(frame, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kImproveFace:
      cv::cvtColor(frame, face_detect_grey, cv::COLOR_BGR2GRAY, 0);
      detectFaces(face_detect_grey, faces);
      improveFace(frame, faces, show);
      cv::imshow(kFilterTitle, show);
      break;
    case kColorKey:
    default:
      // std::cout << "Displaying Color Image!" << std::endl;
      frame.copyTo(show);
      cv::imshow(kFilterTitle, show);
    }

    // see if there is a waiting keystroke
    char new_key = cv::waitKey(10);
    if (new_key >= 0) {
      // user actually input something
      if (new_key == kSaveImage) {
        if (cv::imwrite(img_save_path, show)) {
          std::cout << "Image saved successfully at " << img_save_path
                    << std::endl;
        } else {
          std::cerr << "Failed to save the image!" << std::endl;
        }
      } else {
        // not saving key, possibly a new command
        last_key = new_key;
      }
    }

    if (kQuitKey == last_key) {
      break;
    }
  }

  camera.release();
  return 0;
}
