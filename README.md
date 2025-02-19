# My Adventures With [PyTorch](https://pytorch.org)

<img src="https://pytorch.org/assets/images/logo.svg" style="width:70%; display:block; margin-left:auto; margin-right:auto">

**This repo is under construction**

Curriculum by [Daniel Bourke](https://www.learnpytorch.io/). These are just my notes according to the author of the original content creator along with improvements and PyTorch changes as of January 2025.

## Table of Contents

1. [What is PyTorch?](#what-is-pytorch)
2. [Installing PyTorch](#installing-pytorch)
   1. [How Did I Installed?](#how-did-i-installed)
3. [When in Doubt](#when-in-doubt)
4. [Representing the Data: Tensors](#representing-the-data-tensors)
5. [Approximation to a Solution: Models](#approximation-to-a-solution-models)
6. [Predicting Discrete Values: Classification](#predicting-discrete-values-classification)
7. [Computer Vision with torchvision](#computer-vision-with-torchvision)
8. [Training with Custom Datasets](#training-with-custom-datasets)

## What is PyTorch?

PyTorch is a fullstack ML/DL framework that can handle preprocessing of data and create, train, test models as well as deploy models to an application or cloud. Originally designed by Meta then open-sourced. GPU and TPU acceleration is out of the box via CUDA for NVidia and ROCm for AMD.

## Installing PyTorch

**Locally:**

1. Go to [PyTorch Getting Started](https://pytorch.org/get-started/locally/).
2. Choose your preferences.
3. You will be guided.

**On Cloud:**

- Use Google Colab or Kaggle Notebooks. Necessary packages are already installed there.

### How did I Installed?

1. Locally on Windows 10 machine. Because cloud providers have limits on GPU usage. I prefer test on local then upload code to cloud and do actual heavy training.

2. Version 2.5.1 (latest verion of the time).

3. Download [CUDA](https://developer.nvidia.com/cuda-12-4-0-download-archive) 12.4 for NVidia GPU support.

4. Select **Custom (Advanced)** at **Options**. There are 4 boxes all ticked. One for CUDA tools, one for Geforce Experience app and two for drivers. If your GPU drivers are up to date untick last two boxes. Untick the Geforce Experience box too since it is deprecated. **TL;DR** If you are keeping your GPU drivers up to date Select only CUDA box.

5. Create a python virtual environment with desired python version at a folder using terminal (ps or cmd). Used Python 3.11. `python<version> -m venv <env-name>`

   ```cmd
   python3.11 -m venv .venv
   ```

6. After virtual environment created go to Scripts and run activate script.

   ```cmd
   cd ./.venv/Scripts
   ```

   ```cmd
   activate
   ```

   or just

   ```cmd
   .venv/Scripts/activate
   ```

7. Python with pip inside a virtual environment. [TODO] (See setting up a python virtual environment).

   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

**Note:** Above command would install the latest version of PyTorch. If you want to install a specific version, refer to [here](https://pytorch.org/get-started/previous-versions/).

And done.

## When in Doubt

**Documentation is the king**. Refer to the [docs](https://pytorch.org/docs/)

## Representing the Data: [Tensors](TENSORS.md)

Computers can only understand numbers. Numbers Mason, what do they mean?

**Topics covered:**

1. What is a Tensor?
2. Creating Tensors
3. Tensor Properties
4. Tensor Operations
   - Basic Mathematical Operations
   - Matrix Multiplication
   - Aggregation
   - Shape Manipulation
5. Indexing
6. Tensors and Numpy
7. Reproducibility
8. Tensors on GPU

## Approximation to a Solution: [Models](MODELS.md)

In this chapter, introduction to model operations utilized for training a regression model.

**Topics Covered:**

1. Model Description
2. Mathematical Model and Model in Machine Learning
3. Creating a Regression Model in PyTorch
4. Inference with Model
5. Inside a Model
6. Model Training
   1. Forward pass
   2. Gradient Descent
   3. Loss Function
   4. Optimizer
   5. Backpropagation
   6. Train Loop and Test Loop

## Predicting Discrete Values: [Classification](CLASSIFICATION.md)

**Topics Covered:**

1. [Binary Classification](CLASSIFICATION.BINARY.md)
2. [Multiclass Classification](CLASSIFICATION.MULTICLASS.md)

## [Computer Vision](computer-vision.ipynb) with torchvision

**Topics Covered:**

1.  `torchvision` package for computer vision applications
2.  Built-in datasets
3.  Transforming data to tensor with Transforms API v2
4.  Custom data loading pipeline - DataLoader
5.  Dataset batching
6.  Reusable training loop
7.  Convolutional Neural Network based on TinyVGG
8.  Confusion Matrix
    1. Calculating with `torchmetrcis`
    2. Visualizing with `mlxtend`

## Training with [Custom Datasets](custom-datasets.ipynb)

In previous examples, data was simple enough to fit in memory and working specifically for PyTorch. In this chapter, shown how to import an external structured dataset for performing multiclass image classification.

**Topics Covered:**

1. Dataset import with `torchvision.datasets.ImageFolder`
2. Converting PIL images to tensors with Transforms API v2
3. Creating a batched dataset with `torch.data.DataLoader`
4. Convolutional Neural Network based on TinyVGG
5. Reusable training loop
6. Plotting results
7. Data augmentation with `TrivialAugmentWide`
