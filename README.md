# Fruit and Vegetable Recognition Model

This repository contains the code and data for a **Fruit and Vegetable Recognition** model built using a Convolutional Neural Network (CNN). The model is designed to classify different types of fruits and vegetables based on image data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project is focused on identifying different types of fruits and vegetables using deep learning techniques. The model utilizes image classification to differentiate various categories of produce. The code for testing the trained model is provided in the file `testing_fruit_veg_recognition.py`.

## Dataset
The dataset used for training and testing is obtained from [source of the dataset] (insert dataset source). It contains multiple labeled images for each class (e.g., apples, bananas, carrots, etc.).

The dataset is preprocessed using image augmentation techniques to improve the generalization of the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fruit-veg-recognition.git
   cd fruit-veg-recognition
## Usage 
1. Load the model :- tarined_model.h5
2. Change :- The path of the dataset and model according to you in google colab

## Model Architecture 
The model is built using a Convolutional Neural Network (CNN) architecture, consisting of the following layers:

1. Convolutional layers: Extract features from the images using filters.
2. Pooling layers: Reduce the spatial dimensions of the feature maps to prevent overfitting.
3. Fully connected layers: Combine the extracted features to classify the input into one of the predefined categories.
