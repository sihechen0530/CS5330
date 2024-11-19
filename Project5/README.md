# CS 5330 Project 5: Recognition using Deep Networks
# Description
In this project, we utilize pytorch to train deep network to do various recognition tasks. We trained a basic network structure on the MNIST dataset and test it out on handwritten dataset. Then, we visualized the first layer of the network and its effect on the original image. After that, we used a small Greek letter dataset for transfer learning, essentially replacing the classifier and trying to find out how the network performs. Finally, we run experiments on multiple combination of hyper parameters and check the influence of each hyper parameter on the result.
# Project Highlights:
* Digit recognition on new hand written data
* Automated training on a set of different hyper-parameters
# Environment Setup
* Linux martinxps 5.15.153.1-microsoft-standard-WSL2 #1 SMP Fri Mar 29 23:14:13 UTC 2024 x86_64 x86_64 x86_64 GNU/LinuxOpenCV version: 4.10.0-dev
* Python 3.12.7
* PyTorch version: 2.5.1+cu124
* CUDA version: 12.4
* NumPy version: 2.1.2
* Torchvision version: 0.20.1+cu124
* Matplotlib version: 3.9.2
* GPU (optional for acceleration)
## Contact
Sihe Chen (002085773) chen.sihe1@northeastern.edu
## TODO
* greek letter recognition needs to be improved
* inverstigate failed experiment on some hyper-paremeters
