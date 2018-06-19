# **Build a Traffic Sign Recognition Classifier** 

## Overview
---

This repository is for the project of **Udacity Nanodegree - Self-driving Car Engineer : Build a Traffic Sign Recognition Classifier**.  )It is forked from https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project).  


The goal of this project is to classfy German traffic signs into 43 classes.  

Important concepts used here are Convolutional Network and LeNet. the basic LeNet network architecture was used, and during iterative process some hyper-parameters were tuned. 

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.3%
* test set accuracy of 92.4%

But, I couldn't get good results from traffic sign images found on web. It will be tuned soon.
---
After applying batch normalization in my network, accuracy has been improved:
* Train Accuracy = 1.000
* Validation Accuracy = 0.958
* Test Accuracy = 0.933
And, with my test images, Lenet classfies correctly 4 out of 5 images at most (it normally has 40% accuracy with test images).


## Outputs
---
This project has 3 types of outputs:
1. Ipython Notebook with code / HTML file
2. writeup.md : Description of algorithms and process used in this project. 
