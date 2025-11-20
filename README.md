# Stereo Camera Calibration with OpenCV
This is a project using stereo camera for calibration with OpenCV - C++ based.


## Goal
The main goal of this method is with stereo camera, we can get the dimension, depth, roll, pitch, yaw of the object from 2D images coordinates to 3D world coordinates.


## Setup Instructions
1. Dataset Preparation
In this project, i use 2 iphone cameras as stereo camera. You can use any 2 cameras you have. This tip is for those who don't have stereo camera.
Record a video using your 2 cameras simultaneously. Record a video of chessboard pattern from different angles and distances. After recording, extract frames from the video using any video to frames tool. Save the frames in folders named "left" and "right" for left and right camera images respectively.
After preparing the dataset of chessboard images for calibration, you will need to extract frames for object detection too. So you can use the same method to extract frames from videos where you want to perform object detection.

2. Calibration
With the dataset ready, you can proceed to calibrate the stereo camera using OpenCV. The calibration process involves detecting the chessboard corners in the images and computing the camera parameters.
You can use the provided C++ code to perform stereo calibration. Make sure to adjust the chessboard dimensions and square size according to your chessboard pattern.

3. Object Detection
After calibration, you can use the stereo camera setup for object detection. You can implement object detection algorithms using OpenCV or integrate pre-trained models for better accuracy. Then with the datas of object, and with the params of stereo camera, you can compute the dimension, depth, roll, pitch, yaw of the object.

## Demo
Youtube Link: [Stereo Camera Calibration and Object Detection Demo](https://youtu.be/Nd_e1mUyfyY)

