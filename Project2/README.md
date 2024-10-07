# CS 5330 Project 2: Content-based Image Retrieval
# Description
In this project, we implemented many image feature extractor functions and metrics calculator functions to find matching image in the database. We also introduced an extensible framework that allows developers to add more methods and run these methods using Json configuration files. 
**Currently supported feature extraction methods are:**
1. region of interest pixel extraction (roi)
2. rg chromaticity space histogram (rgHistogram)
3. rgb color histogram (rgbHistogram)
4. sobel magnitude and angle histogram (sobelHistogram)
5. laws filter histogram (laws)
6. gabor filter histogram (gabor)
7. fourier transform histogram (fourier)
**Currently supported metric calculator methods are:**
1. sum of squared distance (SSD)
2. histogram intersection (HistIntersection)
3. cosine distance (cosine)
4. cross entropy (crossEntropy)
5. L-infinity distance (LN)
The keywords in parentheses above can be used to designate method in configuration file.
# Project Highlights:
1. We use a `VectorDatabase` class to store and manage all the features. The feature will be automatically load from file during initialization and saved to files on exit. If the queried feature exists in the database, it won't be computed again during runtime.
2. We use config files to represent each task configurations, including supporting using multiple feature extraction methods on different region of the image at one time, and designating metric calculation method.
3. (TODO) Support parallelized computation of features using OpenMP to accelerate the first computation of features.
# Environment Setup
* WSL2: `Linux LEGION-MARTIN 5.15.153.1-microsoft-standard-WSL2+ #1 SMP Sat Sep 14 13:55:47 PDT 2024 x86_64 x86_64 x86_64 GNU/Linux`
* Windows 11: Enterprise Version, 23H2, 22631.4168
* OpenCV version: 4.10.0-dev
* Compiler version (gcc/g++): 11.4.0
## How to Build and Run
* `cmake -B build && cd build && make && ./imageMatch <directory> <config>`
* see config example file in config/sample_config.json
* compilation products are stored in `build` so that they won't be mixed in code files.
## How to contribute
* develop code in feature_extractor and metric_calculator
* register the function in the `unordered_map` in the top of the file
* provide example for configuration file
## Contact
Sihe Chen (002085773) chen.sihe1@northeastern.edu
