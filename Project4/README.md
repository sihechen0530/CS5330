# CS 5330 Project 4: Calibration and Augmented Reality
# Description
In this project, we use OpenCV functions to calibrate a camera with a chessboard setup and get its intrinsic parameters, after which we calculate the pose of the camera and render objects on the image frame in real time. We then explored the possibility of using Harris Corners as feature points for calibration.
# Project Highlights:
* Realtime video stream processing
* Simple steps for camera calibration and object projection
# Environment Setup
* 20.04.6 LTS (Focal Fossa): `Linux ubuntu 5.15.0-124-generic #134~20.04.1-Ubuntu SMP`
* OpenCV version: 4.10.0-dev
* Compiler version (gcc/g++): 11.4.0
* streaming software: iriun webcam (the phone and the computer have to be in the same Wifi)
* a chessboard
## How to Build and Run
* `cmake -B build && cd build && cmake .. && make && ./corner_detection`
* compilation products are stored in `build` so that they won't be mixed in code files.
## Contact
Sihe Chen (002085773) chen.sihe1@northeastern.edu
## TODO
* Implementing 3D object rendering with OpenGL
