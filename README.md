# Image-classification-with-MobileNet


## Project Description
This project implements a Convolutional Neural Network (CNN) for classifying images of cats and dogs.  
The model uses `TensorFlow` and `Keras` and is trained on a subset of the [Cats and Dogs dataset](https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip).

Key features:
- Image preprocessing with `ImageDataGenerator`
- Data augmentation for training
- CNN architecture using `MobileNet` as a base
- Training with `categorical_crossentropy` loss and `Adam` optimizer
- Accuracy and loss visualization
- Saved trained model for future inference

---

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Korny998/Image-classification-with-MobileNet.git
cd Image-classification-with-MobileNet

```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. **Download and preprocess the dataset and train the CNN model:**

```bash
python train.py
```

2. **You can visualize a batch of confirmation images:**

```bash
python image_example.py
```

## Model Architecture

The CNN model consists of the following layers:

1. Base model: MobileNet (pretrained on ImageNet, top excluded)

2. Input layer: 150x150x3

3. GlobalAveragePooling2D layer

4. Dense layer: 64 units, ReLU activation

5. Dropout layer: 0.5

6. Output Dense layer: 2 units, Softmax activation (cats vs dogs)

All layers of the base MobileNet are frozen during training.

## Dataset

The dataset is downloaded automatically from:

```bash
https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip
```

The dataset.py script splits the dataset into:
  - Train: 2800 images per class
  - Validation: 600 images per class
  - Test: 600 images per class

The images are organized into separate folders for cats and dogs in each split.