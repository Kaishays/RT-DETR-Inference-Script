# RT-DETR Inference Script

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This repository contains the inference script for the custom **RT-DETR** (Real-Time Detection Transformer) model designed to detect cars. The RT-DETR model leverages a transformer architecture, offering improved detection capabilities over traditional motion estimator models.

## Video Demonstration

[![RT-DETR vs Commercial Model](https://img.youtube.com/vi/DrHrfP9c5uE/0.jpg)](https://youtu.be/DrHrfP9c5uE)

*This video compares the custom RT-DETR model I trained to detect cars with a currently available commercial model. In the video, the RT-DETR model is displayed on the left, and the commercial model is shown on the right. This comparison highlights the advantage of using a transformer architecture over a motion estimator model, as my model can detect classes, stationary objects, and has greater accuracy. However, one advantage of motion estimator models is their ability to detect small objects within a frame. To improve my model's performance in detecting small objects, I plan to implement an algorithm like SAHI (Slicing Aided Hyper Inference) and develop an IoU algorithm to eliminate overlapping bounding boxes.*

## Features

- **Transformer-Based Architecture**: Utilizes transformer models for enhanced detection capabilities.
- **High Accuracy**: Detects classes and stationary objects with greater precision.
