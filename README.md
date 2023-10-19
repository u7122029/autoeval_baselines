# Can Effective Invariance Metrics Predict Classifier Accuracy on a Data-Centric View?

This is a fork of [the 1st DataCV Challenge](https://sites.google.com/view/vdu-cvpr23/competition?authuser=0)

## Table of Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Related Work](#related-work)
- [Datasets](#datasets)
    - [Interior Domain](#interior-domain)
    - [Exterior Domain](#exterior-domain)
    - [Downloads](#downloads)
- [Method](#method)
- [Code Execution](#code-execution)
- [Self-Supervised Task Metrics](#self-supervised-task-metrics)
    - [Rotation](#rotation)
    - [Jigsaw](#jigsaw)
- [Results](#results)
    - [Interior Domain](#interior-domain-1)
    - [Exterior Domain](#exterior-domain-1)
    - [Exterior Domain With Interior Domain Fit](#exterior-domain-with-interior-domain-fit)

## Abstract

We investigate the relationship between image classification accuracy (classification) and jigsaw invariance (JI) over the 
CIFAR-10 dataset and its variants. This provides extra insight into using metrics derived from Effective Invariance (EI) to predict classification accuracy
where labels for a given dataset are not provided to help determine a classifier's performance over it. We also compare the strength of the correlation between classification and JI with other metrics such as rotation invariance (RI), rotation prediction (rotation), jigsaw prediction (jigsaw) and the nuclear norm. We find that there is no clear
correlation between classification and JI, which is consistent with classification vs. RI. Most models showed inconsistent results over both metrics, while the correlation was more consistent but moderately strong with rotation and jigsaw as the independent variables. Meanwhile the correlation was strongest with the nuclear norm as the independent variable, which could be made stronger with the natural logarithm taken on classification. This overall shows that a model's classification accuracy cannot be reliably predicted using JI or RI, but rather with the nuclear norm.


<!--We also verify the same correlation with 
jigsaw solving accuracy (jigsaw) in place of rotation. For both comparisons, all models (except one) showed a strong 
correlation over datasets constructed by directly using image transformations ($R^2 > 0.71$ for classification vs. 
rotation, $R^2 > 0.61$ for classification vs. jigsaw). However, some models showed a weak linear correlation between 
classification and rotation/jigsaw accuracy on images outside of CIFAR-10. This suggests that the effectiveness of 
estimating a model's classification accuracy based on rotation/jigsaw accuracy depends on the model itself, however, 
this must be investigated further.-->

## Overview

Constructing a dataset of images for a supervised task such as classification requires generating labels for every
image.
Each image must be manually labelled by humans. Furthermore, the most complex computer vision machine learning models
require tens of thousands, perhaps even millions of images to train on. This makes labelling images a time-consuming,
repetitive and laborious task with room for human error.

One way to automate this task is to computationally generate labels for these images, however, current machine learning
models are generally not as accurate as the human brain at identifying images. Computers on the other hand are generally
good at performing deterministic algorithms such as image transformations which include rotations, jigsaw-making 
(dividing the image into a grid of squares and rearranging them, hence constructing a jigsaw puzzle), blurring, 
sharpening, and many more operations.

In this way, we can ask ourselves whether there is a correlation of some form between a model's performance at a
supervised task such as image classification and said model's performance at some self-supervised or unsupervised task, 
such as
- rotation prediction - predicting the orientation of an image,
- jigsaw puzzle solving - predicting the permutation of a jigsaw-made image,
- rotation invariance - a comparison between image classification on regular images and their rotated counterparts,
- jigsaw invariance - the same comparison but for jigsaw-made images,
- nuclear norm - using the sum singular values of the compiled softmax prediction vectors.

For rotation prediction and jigsaw puzzle solving, labels can easily be generated accurately by a computer without 
human intervention - hence the name "self-supervision". The other tasks do not require image labels, so these are known 
as "unsupervised" tasks. If there is a correlation between image classification accuracy and any of these tasks, we can 
then estimate a model's classification accuracy given its performance on any of these tasks.
In a broader sense, it will be much easier to verify if a given supervised task can have a model fitted to it with 
sufficiently high performance - all before any time or human resources are spent generating labels.

## Datasets

### For Pretrained Models

Every model was trained on the original [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for
classification
without any transformations applied to each image aside from standardisation.

### Interior Domain

The interior domain contains 2400 transformed datasets from the original
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test set, using the transformation strategies proposed by
[Deng et al. (2021)](https://github.com/Simon4Yan/Meta-set).

### Exterior Domain

The exterior domain contains datasets from

- [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) (1 dataset),
- CIFAR-10.1-C - the original CIFAR-10.1 dataset but with image transformations from [(Hendrycks et. al., 2019)](https://github.com/hendrycks/robustness) applied (95 datasets)
- [CIFAR-10.2](https://github.com/modestyachts/CIFAR-10.2)
- CIFAR-10.2-C - the original CIFAR-10.2 dataset but with image transformations from [(Hendrycks et. al., 2019)](https://github.com/hendrycks/robustness) applied (95 datasets)
- CIFAR-10-F-32 - real-world images collected from [Flickr](https://www.flickr.com) compiled by [(Sun et. al., 2019)](https://github.com/sxzrt/CIFAR-10-W#cifar-10-warehouse-towards-broad-and-more-realistic-testbeds-in-model-generalization-analysis), resized to 32 by 32 images. (20 datasets)
- CIFAR-10-F-C - uses the exact same images as CIFAR-10-F-32 but with image transformations from [(Hendrycks et al., 2019)](https://github.com/hendrycks/robustness) applied (1900 datasets)
- CIFAR-10-W - Datasets obtained from various sources as well as through diffusion models.

The CIFAR-10.1 dataset is a single dataset, while CIFAR-10.1-C and CIFAR-10-F contain 19 and 20 datasets respectively.
Therefore, the total number of datasets in the exterior domain is 40.

For every dataset in the exterior domain, the image file and their labels are stored as two separate `Numpy` array files
named "data.npy" and "labels.npy". The PyTorch implementation of the Dataset class for loading the data can be found
in `utils.py`.

### Downloads

Download the interior domain
datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Eb9yO_Qg41lOkoRS7P6gmqMBk5Q6A2gCV8YbRbuLpB8NwQ?e=WO3Gqi)

Download the exterior domain
datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Edg83yRxM9BPonPP22suB_IBrHlKYV5bOn4VK-c5RZ8dtQ?e=kExXEm)

## Method

Without delving into too much technical detail, we can summarise our method as follows for a selected model:

1. Load pretrained weights of model,
2. Evaluate model's performance on image classification over the interior and exterior domains,
3. Train model's self-supervision layer on rotation/jigsaw prediction using the original CIFAR-10 dataset for a maximum
   of 25 epochs with learning rate `1e-2`,
4. Evaluate model's performance on rotation/jigsaw prediction using the datasets in the interior and exterior domains,
5. Fit linear regression model for image classification accuracy vs rotation/jigsaw prediction accuracy.

## Pretrained Model Weights

Each model we tested had pretrained weights for image classification from an exterior source. This ensures near-optimal
performance on image classification. We list the source, followed by the model names.

From [akamaster/pytorch_resnet_cifar10](https://github.com/u7122029/pytorch_resnet_cifar10):

- ResNet-110,
- ResNet-1202

From [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models/tree/master):

- All other versions of ResNet,
- MobileNet_v2,
- RepVGG,
- ShuffleNet

From [huyvnphan/PyTorch_CIFAR10](https://github.com/u7122029/PyTorch_CIFAR10):

- All versions of DenseNet,
- Inception_v3,
- LeNet5,
- Linear

We made forks of each repository listed to upload the pretrained weights to the releases section, and we give full
credit to the original owners of the uploaded files where specified. Please note however, that some pretrained weights
were produced by us, specifically in the [u7122029/pytorch-cifar10](https://github.com/u7122029/pytorch-cifar10)
repository.

## Code Execution

To install the required Python libraries, please execute the command below.

```bash
pip3 install -r requirements.txt
```

To test our results on one self-supervised task, please download and unpack the datasets from the
[downloads](#downloads) section into `code/data` and use the following.

```bash
cd code
python3 main.py
```
This will run all classifiers based on the settings given in `test_config.json`.

### Self-Supervised Task Metrics

We present the metrics for each self-supervised task used in this repository below.

#### Rotation

The *Rotation Prediction* (Rotation) [(Deng et al., 2021)](https://arxiv.org/abs/2106.05961) metric is defined as

```math
\text{Rotation} = \frac{1}{m}\sum_{j=1}^m\Big\{\frac{1}{4}\sum_{r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}}\mathbf{1}\{C^r(\tilde{x}_j; \hat{\theta}) \neq y_r\}\Big\},
```

where

- $m$ is the number of images,
- $y_r$ is the label for $r \in \lbrace 0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ} \rbrace$,
- $C^r(\tilde{x}_j; \hat{\theta})$ predicts the rotation degree of an image $\tilde{x}_j$.

#### Jigsaw

The *Jigsaw Prediction* (Jigsaw) metric is defined as

```math
\text{Jigsaw} = \frac{1}{m}\sum_{j = 1}^m\left(\frac{1}{g!}\sum_{p \in [1,g!] \cap \mathbb{N}}\mathbf{1}_{\{x | C^p(x;\hat{\theta}) \neq y_p\}}(\tilde{x}_j)\right)
```

where

- $m$ is the number of images,
- $g$ is the length and width of the jigsaw puzzle grid,
- $y_r$ is the label for $r \in [1,g!] \cap \mathbb{N}$,
- $C^p(\tilde{x}_j; \hat{\theta})$ predicts the permutation index of a jigsaw image $\tilde{x}_j$.

## Results

The tables in each section below show the results of the linear models fitting to their corresponding domains using the
[Root-Mean-Square Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE), and the
[Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) ($R^2$ Coefficient).

All tables are sorted in descending order by $R^2$ coefficient.

