# U-Net: Convolutional Networks for Biomedical Image Segmentation

This repository contains the implementation of the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. This implementation includes the U-Net model and training scripts for biomedical image segmentation.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenizer](#tokenizer)
  - [Training the Transformer](#training-the-transformer)
- [Examples](#examples)
- [References](#references)

## Overview

The U-Net architecture is designed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. This makes it particularly effective for tasks where a detailed understanding of the input image is necessary. In the original paper they don't use layer normalization nor batch normalization, but as this techniques make training more stable and helps to achieve convergence faster, we decided to include torch LayerNorm module after each convolution layer.

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
   git clone https://github.com/deeplearningcafe/transformer-pytorch.git
   cd transformer-pytorch
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

### Hyperparameter search
By overfitting in a single batch, we can check if the model is working well if it can achieve 0 loss. To run it just change the `hp_search` variable to True inside the `config.yaml` file, the tolerance, max steps and search interval can be changed.
   ```bash
   python training.py
   ```

### Training the U-Net Model
To train the Transformer model, use the provided script:
   ```bash
   python training.py
   ```
Here, `config.yaml` is a configuration file specifying the model parameters, training settings, and dataset paths. Parameter count becomes 31,030,658 while try to use the same configuration as in the paper.
   ```yalm
unet:
  input_channels: [64, 128, 256, 512]
  crop_size: [56, 104, 200, 392]
  num_classes: 2
  image_channels: 1
  dropout: 0.1

train:
  data: 'data'
  batch_size: 1
  scheduler_type: 'warmup-cosine'
  max_epochs: 100
  warmup_epochs: 10
  use_bitsandbytes: True
  optim: 'sgd'
  lr: 1e-3
  device: 'cuda'
  save_path: 'weights'
  eval_epoch: 1
  log_epoch: 1
  save_epoch: 1
   ```

## Examples
We include the `inference.py` file for inference.

## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint [arXiv:1505.04597.](https://arxiv.org/abs/1505.04597)
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License
This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file. I don't own the dataset, its license can be found in the EM segmentation challenge.
