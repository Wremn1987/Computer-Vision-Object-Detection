# Computer Vision Object Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red?style=flat-square&logo=pytorch)](https://pytorch.org/)

This repository contains implementations and experimental setups for various state-of-the-art object detection models. The goal is to provide a practical toolkit for applying object detection in real-world scenarios, including custom dataset training and deployment considerations.

## Table of Contents
- [Introduction](#introduction)
- [Supported Models](#supported-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Custom Dataset Training](#custom-dataset-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Object detection is a fundamental task in computer vision that involves identifying and localizing objects within an image or video. This project focuses on popular and effective models like YOLO (You Only Look Once) and Faster R-CNN, demonstrating their application and fine-tuning.

## Supported Models
- **YOLOv5/YOLOv8:** Real-time object detection.
- **Faster R-CNN:** Two-stage detector for higher accuracy.
- **SSD (Single Shot Detector):** Balance between speed and accuracy.

## Project Structure
```
.gitignore
README.md
requirements.txt
src/
├── __init__.py
├── yolo_detector.py
└── faster_rcnn_detector.py
data/
├── images/
└── annotations/
configs/
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Wremn1987/Computer-Vision-Object-Detection.git
   cd Computer-Vision-Object-Detection
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- To run YOLO detection on an image:
  ```bash
  python src/yolo_detector.py --image_path data/images/test.jpg --weights yolov5s.pt
  ```
- To run Faster R-CNN detection:
  ```bash
  python src/faster_rcnn_detector.py --image_path data/images/test.jpg --config configs/faster_rcnn.yaml
  ```

## Custom Dataset Training
Instructions for preparing custom datasets and training models can be found in the `docs/` directory (to be added).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
