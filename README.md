# Real-time Driver Drowsiness Detection

## Overview

This project implements a real-time drowsiness detection system using computer vision and facial landmarks analysis. The system is designed to monitor user attentiveness and issue immediate audio alerts for drowsy and sleeping states. It aims to enhance safety in scenarios where sustained attention is critical, such as driving or operating machinery.

## Features

- **Facial Landmarks Analysis:** Utilizes the dlib library for detecting faces and extracting facial landmarks from the video feed.

- **Eye Aspect Ratio (EAR):** Calculates the EAR to analyze blink patterns, a key indicator of drowsiness.

- **Audio Alerts:** Integrates Pygame for playing customizable audio alerts when drowsiness or sleeping is detected.

- **Dynamic Thresholds:** Adjustable parameters for frame resolution, alert thresholds, and detection intervals to fine-tune system performance.

- **Real-time Visualizations:** Displays real-time visualizations with highlighted detected faces and facial landmarks. Provides status updates (Active, Drowsy, Sleeping) on the video feed.

## Requirements

- Python 
- OpenCV
- dlib
- Pygame
- NumPy
- imutils

## Usage

- Install necessary libraries

- Download the python file and the audio files and place it in a folder.

- Add the 68-landmark-dat file from [here](https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat) and place it in the same folder.

- Ensure that the camera is accessible amd run the python script.


## Configuration

Adjust the following parameters in the script for customization:

- **Frame resolution:** Set the width and height according to your preferences.

  ```python
  cap.set(3, 640)  # Width
  cap.set(4, 480)  # Height

- **Alert thresholds:** Fine-tune the thresholds for blink detection.

  ```python
  threshold = 0.2


## Future Enhancements

- Integration with IoT devices for broader application.

## Output 

![States Detected](https://github.com/Nikhilesh711/Driver-Drowsiness-Detection/assets/100184888/97c7dbbe-7e45-4213-a198-e5378f2fd479)


