# CS5330 Project 1
## Project Description
In this project, we start from installing opencv and displaying a static image, and then integrate live camera video stream into our program, after which we implemented multiple filters and effects on the video frames. The filters include but are not limited to GrayScale filter, Sepia tone filter, Gaussian blur filter, Sobel X/Y/magnitude filter, as well as facial detection features. In the last three self-designed filters, frosted glass filter, time delay effect as well as facial improvement filter are implemented.
## Environment Setup
* WSL2: `Linux LEGION-MARTIN 5.15.153.1-microsoft-standard-WSL2+ #1 SMP Sat Sep 14 13:55:47 PDT 2024 x86_64 x86_64 x86_64 GNU/Linux`
* Windows 11: Enterprise Version, 23H2, 22631.4168
* OpenCV version: 4.10.0-dev
* Compiler version (gcc/g++): 11.4.0
* compilation products are stored in `build` so that they won't be mixed in code files.
## How to Build and Run
* `cmake -B build && cd build && make && cd .. && ./build/vidDisplay <save_img_path>`
## Usage
| Key | Function |
| --- | ------|
| q | Quit |
| p | Save the current frame |
| g | Convert Image to GrayScale |
| h | Convert Image to GrayScale (a different version of implementation) |
| s | Sepia Tone effect |
| c | get colored image (default) |
| b | blur the image using Gaussian filter |
| x | apply Sobel X filter |
| y | apply Sobel Y filter |
| m | compute sobel magnitude and display |
| l | blur + quantize the image |
| f | detect face and show the bounding box |
| d | frosted glass distortion effect |
| t | time delay effect |
| i | improve face detection |
## Contact
Sihe Chen (002085773) chen.sihe1@northeastern.edu
