# CS 5330 Project 3: Real-time 2-D Object Recognition
# Description
In this project, we utilize a streaming platform to stream phone camera video to our program in realtime, processing frame by frame with OpenCV. We start by turning the image into a grayscale one, thresholding it and segmenting the image into regions, and then we compute moments of regions as features to identify the object in each region, storing the features in a database for future matching. User can easily annotate the object by clicking on the box and the system will keep a record for future matching task.
# Project Highlights:
* Realtime video stream processing
* Use K-means (K=2) to calculate the threshold to turn the grayscale image into a binary one.
* One-step annotation and matching system
# Environment Setup
* 20.04.6 LTS (Focal Fossa): `Linux ubuntu 5.15.0-124-generic #134~20.04.1-Ubuntu SMP`
* OpenCV version: 4.10.0-dev
* Compiler version (gcc/g++): 11.4.0
* streaming software: iriun webcam (the phone and the computer have to be in the same Wifi)
## How to Build and Run
* `cmake -B build && cd build && cmake .. && make && ./2DRecog`
* compilation products are stored in `build` so that they won't be mixed in code files.
## Contact
Sihe Chen (002085773) chen.sihe1@northeastern.edu
## Travel Days
travel days used for this project: 2
