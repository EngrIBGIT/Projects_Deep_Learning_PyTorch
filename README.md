# Projects_Deep_Learning_PyTorch

## Deep Learning with PyTorch

Welcome to my **Deep Learning with PyTorch** repository! This project serves as a comprehensive guide to my PyTorch learning , starting from the basics and advancing to topics like Neural Networks, Artificial Neural Networks (ANN), and Convolutional Neural Networks (CNN). 

This repository is designed as a help to master PyTorch and its powerful applications.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installing PyTorch](#installing-pytorch)
  - [Setting up the Environment](#setting-up-the-environment)
- [Learning PyTorch Fundamentals](#learning-pytorch-fundamentals)
  - [Tensors in PyTorch](#tensors-in-pytorch)
  - [Autograd in PyTorch](#autograd-in-pytorch)
  - [PyTorch Modules](#pytorch-modules)
- [Deep Learning Concepts](#deep-learning-concepts)
  - [Neural Networks Basics](#neural-networks-basics)
  - [Artificial Neural Networks (ANN)](#artificial-neural-networks-ann)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview

Deep Learning has transformed fields like computer vision, natural language processing, and autonomous systems. This repository is guide through the essential concepts of deep learning using PyTorch, one of the most popular frameworks.

You will explore:
- PyTorch basics and fundamentals: **tensors**, **autograd**, and **neural network modules**.
- Building **Artificial Neural Networks (ANNs)**.
- Mastering **Convolutional Neural Networks (CNNs)** for image recognition tasks.
- Hands-on examples and exercises to reinforce learning.

By the end, it is to have a solid foundation in PyTorch and the ability to develop, train, and deploy deep learning models.

## Prerequisites

To make the most out of this repository, it is recommended to have:
- Basic knowledge of Python programming.
- Familiarity with linear algebra and calculus.
- Some understanding of machine learning concepts.

The repository includes easy-to-follow instructions and exercises to get you started.

## Getting Started

### Installing PyTorch

Install PyTorch by running:

`pip install torch torchvision `

## Setting up the Environment

It's recommended to use a virtual environment to manage dependencies for this project:

      `I am using` [Google Colab](https://colab.research.google.com)

# Create a virtual environment

`python -m venv pytorch_env`


# Activate the virtual environment
# On Windows:

`pytorch_env\Scripts\activate`

# On MacOS/Linux:

`source pytorch_env/bin/activate`

# Install required dependencies

`pip install -r requirements.txt`



## Learning PyTorch Fundamentals

Tensors in PyTorch

Tensors are the core building blocks of PyTorch models. 

They are multi-dimensional arrays, similar to NumPy arrays.

`import torch`

# Create a simple tensor

`x = torch.tensor([[1, 2], [3, 4]])
print(x)`


## Autograd in PyTorch

Autograd allows automatic differentiation of tensor operations. This is crucial for training neural networks.

`x = torch.tensor(1.0, requires_grad=True)
y = 2 * x
y.backward()
print(x.grad)  # Outputs the gradient of y with respect to x`

I am using` [Google Colab](https://colab.research.google.com)

## PyTorch Modules
PyTorch uses modules (torch.nn.Module) to represent models, enabling the creation of complex architectures like neural networks.

### Deep Learning Concepts

**Neural Networks Basics**

A Neural Network consists of multiple layers that process data to make predictions. 

PyTorch allows for easy building of such networks using torch.nn.Module.

`import torch.nn as nn`

`class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # Fully connected layer

    def forward(self, x):
        return self.fc1(x)`


## Artificial Neural Networks (ANN)

Artificial Neural Networks are the foundation of deep learning. 

*Learning:* [Deep Learning with PyTorch Notebook](https://github.com/EngrIBGIT/Projects_Deep_Learning_PyTorch/blob/main/Deep_Learning_Pytorch.ipynb)


Build basic ANN models in PyTorch.

Train models using gradient descent.

Fine-tune parameters for improved performance.


## Convolutional Neural Networks (CNN)

Convolutional Neural Networks are designed for image processing tasks. 

*Learning:*[Convolutional Neural Network PyTorch Notebook](https://github.com/EngrIBGIT/Projects_Deep_Learning_PyTorch/blob/main/Convolusional_Neural_Network_Pytorch.ipynb)


How to build CNN layers with torch.nn.Conv2d.

Using pooling layers to reduce dimensionality.

Training CNNs on datasets like CIFAR-10.

`class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x`

Project Link:

[Deep Learning with PyTorch Notebook](https://github.com/EngrIBGIT/Projects_Deep_Learning_PyTorch/blob/main/Deep_Learning_Pytorch.ipynb)

[Convolutional Neural Network PyTorch Notebook](https://github.com/EngrIBGIT/Projects_Deep_Learning_PyTorch/blob/main/Convolusional_Neural_Network_Pytorch.ipynb)

[Deep Learning with PyTorch Project](https://github.com/EngrIBGIT/Projects_Deep_Learning_PyTorch/blob/main/Project_Deep_Learning_PyTorch.ipynb)



## Project Structure
The repository is structured as follows:

`
├── data/                   # Folder for datasets

├── notebooks/              # Jupyter notebooks 

for tutorials and experiments
├── models/                 # Pretrained models and architecture files

├── src/                    # Source code for neural network models and utilities

├── requirements.txt        # List of dependencies

├── README.md               # Documentation file (this file)`



## Contributing

I welcome contributions to improve this repository! If you'd like to contribute:

- Fork this repository.

- Create a new branch (git checkout -b feature-branch).

- Commit your changes (git commit -am 'Add some feature').

- Push the branch (git push origin feature-branch).

- Create a new Pull Request.

`**Thank you for exploring my Deep Learning with PyTorch! I hope this repository helps with some deep learning and PyTorch.**`