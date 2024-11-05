# Cataract Detection using ZFNet

This project aims to detect cataracts in eye images using ZFNet, a convolutional neural network architecture, implemented in Python with the TensorFlow framework.

## Introduction

Cataract is a common eye condition characterized by clouding of the lens, which leads to impaired vision and can eventually cause blindness if left untreated. Early detection of cataracts is crucial for timely treatment and prevention of vision loss.

Convolutional Neural Networks (CNNs) have shown remarkable performance in various image recognition tasks, including medical image analysis. ZFNet, a variant of CNN, has been successfully used in image classification tasks and can be adapted for cataract detection.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- OpenCV (for image preprocessing)
- Matplotlib (for visualization)
- Dataset of eye images (with annotations if available)

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/cataract-detection-zfnet.git

2. Install dependencies:
pip install -r requirements.txt


## Usage

1. Preprocess the dataset:
- Organize your dataset into training and testing sets.
- If necessary, preprocess the images (e.g., resizing, normalization, augmentation).

2. Train the ZFNet model:
python train.py --dataset path_to_dataset --epochs num_epochs --batch_size batch_size


## Dataset

Ensure that you have a dataset of eye images containing both cataract and normal images. If annotations (labels) are available indicating which images contain cataracts, it will be helpful for training the model.

## Model

ZFNet was designed to improve upon AlexNet by making subtle adjustments to the network's structure, including:

Smaller receptive field: Reduced the size of the first convolutional layer’s filters from 11x11 to 7x7. This change increased feature localization, capturing finer details.

Increased strides: Larger strides in initial layers led to improved data utilization and reduced computational overhead.

Visualization insights: Developed innovative visualization techniques to observe how CNNs interpret images at different layers, a key advancement in understanding the "black-box" nature of neural networks.


These changes led to a more efficient network capable of achieving better accuracy and generalization than its predecessors.

Key Features

Architecture Enhancements: Adjustments to filter sizes, strides, and feature extraction approaches.

Visualization Techniques: Pioneered deconvolutional network visualization to understand the learned features at each layer.

Improved Accuracy: Achieved superior performance on classification tasks compared to AlexNet.


Architecture

ZFNet follows a similar structure to AlexNet, with five convolutional layers followed by three fully connected layers. However, it includes crucial changes to the convolutional layers:

1. First Layer: 7x7 filters with a stride of 2.


2. Subsequent Layers: Various kernel and pooling adjustments to optimize feature extraction.


3. Visualization Layers: Added to inspect intermediate layer outputs, offering insights into how features are learned.



Advantages of ZFNet

Improved localization and recognition: Due to smaller filters in the initial layers.

Reduced overfitting: Better suited to capture high-level image details without excessive complexity.

Enhanced interpretability: Visualization methods provide insights into the hierarchical feature learning of CNNs.

## Results

In the cataract detection project using ZFNet, the model achieved a test accuracy of approximately 89.7% with a test loss of 0.2758. The confusion matrix showed 201 true positives (correctly identified cataract cases), 252 true negatives (correctly identified normal cases), 34 false negatives (cataract cases misclassified as normal), and 24 false positives (normal cases misclassified as cataract). The classification report indicated that for cataract cases, the model achieved a precision of 0.89, recall of 0.86, and F1-score of 0.87, while for normal cases, it achieved a precision of 0.88, recall of 0.91, and F1-score of 0.90. Overall, the model’s balanced accuracy and strong performance metrics demonstrate its effectiveness for cataract detection in medical imaging.


