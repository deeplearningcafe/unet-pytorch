# U-Net: Convolutional Networks for Biomedical Image Segmentation

This repository contains the implementation of the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. This implementation includes the U-Net model and training scripts for biomedical image segmentation.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Transformer](#training-the-transformer)
- [Examples](#examples)
- [References](#references)

## Overview

The U-Net architecture is designed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. This makes it particularly effective for tasks where a detailed understanding of the input image is necessary. In the original paper they don't use layer normalization nor batch normalization, but as this techniques make training more stable and helps to achieve convergence faster, we decided to include torch LayerNorm module after each convolution layer. Also for training in the Penn-Fudan Pedestrian we include padding of value 1 to achieve the same dimensions in the input and the output.

## Requirements

- Python 3.8 or higher
- PyTorch 2.3 or higher
- NumPy
- scikit-image
- tqdm
- matplotlib

## Installation

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/deeplearningcafe/unet-pytorch.git
   cd unet-pytorch
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Or if using conda:
   ```bash
   conda create -n unet_torch
   conda activate unet_torch
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Dataset
We use the dataset used in the paper, from EM segmentation challenge in 2015. This dataset has only 30 images, by using the Overlap-tile as in the paper, from each image we can create 4 samples, also we apply data augmentation. For validation we take 10% of the dataset, that is 3 images.

Also we added the option to train with the [Penn-Fudan Pedestrian](https://www.kaggle.com/datasets/sovitrath/penn-fudan-pedestrian-dataset-for-segmentation), this kaggle version has already been splitted into train and validations sets. The labels can contain several objects, each object is given a pixel starting by 1 for the first object, so we loop all the unique values of the mask and treat all the objects as the same label, so in the end we just have two classes, 0 is background and 1 is person class.

### Hyperparameter search
By overfitting in a single batch, we can check if the model is working well if it can achieve 0 loss. To run it just change the `hp_search` variable to True inside the `config.yaml` file, the tolerance, max steps and search interval can be changed.
   ```bash
   python training.py
   ```
The Unet model using half of the channels, parameter becomes around 7M, achieves a loss of 0.1 when overfitting in one batch.

### Training the U-Net Model
To train the Transformer model, use the provided script:
   ```bash
   python training.py
   ```
Here, `config.yaml` is a configuration file specifying the model parameters, training settings, and dataset paths. Parameter count becomes 31,030,658 while try to use the same configuration as in the paper. In the configuration you can decide if use BatchNorm, the padding and the dataset.
   ```yalm
unet:
  input_channels: [64, 128, 256, 512]
  # input_shape: [570, 282, 138, 66, 30, 54, 102, 198, 390]
  input_shape: [320, 160, 80, 40, 20]
  num_classes: 2
  image_channels: 3
  dropout: 0.1
  use_ln: True
  padding: 1

train:
  data: 'data'
  batch_size: 8
  scheduler_type: 'warmup-cosine'
  max_epochs: 50
  warmup_epochs: 4
  use_bitsandbytes: True
  optim: 'adamw'
  lr: 1e-5
  device: 'cuda'
  save_path: 'weights'
  eval_epoch: 1
  log_epoch: 1
  save_epoch: 3
  log_path: 'logs'
  early_stopping: 50
  w_0: 1.1
  dataset_name: "pedestrian"
   ```
## Training Results
While we tried to use the same settings as in the original paper, we weren't able to achieve convergence using the EM segmentation dataset. Despite including almost the same loss weight as in the paper, that is a sum of class balance and the distance with the nearest cell, around epoch 5 with learning rate of 1e-4 the training loss and the validation loss start to diverge. After several experiments in which we remove the distance loss, we decreased learning rate and change the data augmentation techniques, the results were always the same.

The rotation data augmentation leads too poor results, by taking a look at the training logits it was clear that when the model starts recognizing the rotation, the validation loss tends to raise. So we remove the rotation data augmentation.

For the Penn-Fudan Pedestrian dataset we used the same transformations as in the cells dataset, but in this case the model learned properly, training loss and validation loss were both decreasing.

## Examples
We include the `test.py` file for showing the predictions and check the gradients and activations.

## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint [arXiv:1505.04597.](https://arxiv.org/abs/1505.04597)
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License
This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file. I don't own the dataset, its license can be found in the EM segmentation challenge.
