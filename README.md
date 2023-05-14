# Is the Linear correlation Between Classification and Rotation/Jigsaw Prediction Model-Invariant?

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

We verify that the linear correlation between classification accuracy and self-supervision accuracy for rotation
prediction and jigsaw classification holds between various computer vision neural network models over the CIFAR-10
dataset. We observed that there is a medium to strong linear correlation between classification accuracy and both
self-supervised tasks over both interior (the collection of datasets
derived from the original CIFAR-10 dataset through image transformations) and exterior (the collection of datasets 
sharing labels with the original CIFAR-10 dataset, but containing images not included in the latter) dataset domains.
This suggests that it is not necessary to generate human-made labels for a supervised task since we can gauge model 
performance on said task through its performance on a self-supervised one.

## Overview

Constructing a dataset of images for a supervised task such as classification requires generating labels for every
image.
Each image must be manually labelled by humans. Furthermore, the most accurate computer vision machine learning models
required tens of thousands, perhaps even millions of images to train on. This makes labelling images a time-consuming,
repetitive and laborious task without much obvious potential for automation.

One way to automate this task is to computationally generate labels for these images, however, current machine learning
models are generally not as accurate as the human brain at identifying images. Computers on the other hand are generally
good at performing image transformations such as rotations, jigsaw-making (dividing the image into a grid of squares and
rearranging the squares, hence constructing a jigsaw puzzle), blurring, sharpening, and many more operations.

In this way, we can ask ourselves whether there is a correlation of some form between a model's performance at a
supervised task such as image classification and said model's performance at some self-supervised task, such as rotation
prediction - predicting the orientation of an image - or jigsaw puzzle solving - predicting the permutation of a
jigsaw-made image. For both tasks, labels can easily be generated accurately by a computer without human intervention -
hence the name "self-supervision". If there is a correlation, we can then judge the classification accuracy of a model
given its performance on a self-supervised task. In a broader sense, it will be much easier to verify if a given
supervised task can have a model fitted to it with sufficiently high performance - all before any time or human
resources are spent generating labels.

### Related Work

#### Image Classification - Self-Supervised Task Performance

Aside from the work that this repository was forked from [ref], our work also focuses on the findings of [ref] which
observed a strong linear correlation between a model's performance on image classification and rotation prediction over
MNIST, CIFAR10 and ImageNet, where for each dataset, different models were used. Our work keeps the dataset (i.e:
CIFAR-10) fixed, and we test various models on this dataset to verify that the linear correlation between classification
accuracy and rotation prediction accuracy is preserved. In the future work section of the same paper, the writers also
remark that there is a strong linear correlation between classification accuracy and jigsaw classification accuracy. We
verify this finding in our work as well.

#### CIFAR-10 Dataset

#### ResNet

#### DenseNet

#### MobileNetv2

#### ShuffleNet

#### RepVGG

#### Image Classification

#### Self-Supervised Tasks

## Datasets

### For Pretrained Models

Every model was trained on the original [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for
classification
without any transformations applied to each image aside from normalisation.

### Interior Domain

The interior domain contains 1,000 transformed datasets from the original
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test set, using the transformation strategies proposed by
[Deng et al. (2021)](https://arxiv.org/abs/2007.02915).

The interior domain datasets share a common label file named `labels.npy`, and images files are
named `new_data_xxx.npy`,
where `xxx` is a number from 000 to 999.

### Exterior Domain

The exterior domain contains datasets from

- [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1),
- [CIFAR-10.1-C](https://github.com/hendrycks/robustness) (add
  corruptions [(Hendrycks et al., 2019)](https://arxiv.org/abs/1903.12261)
  to [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) dataset), and
- CIFAR-10-F (real-world images collected from [Flickr](https://www.flickr.com))

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

For [akamaster/pytorch_resnet_cifar10](https://github.com/u7122029/pytorch_resnet_cifar10):

- ResNet-110,
- ResNet-1202

For [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models/tree/master):

- All other versions of ResNet,
- MobileNet_v2,
- RepVGG,
- ShuffleNet

For [huyvnphan/PyTorch_CIFAR10](https://github.com/u7122029/PyTorch_CIFAR10):

- All versions of DenseNet,
- Inception_v3,

For [u7122029/pytorch-cifar10](https://github.com/u7122029/pytorch-cifar10)

- AlexNet,
- LeNet5,
- OBC,
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
cd code/
python3 get_accuracy.py --model MODEL 
python3 baselines/BASELINE.py --model MODEL [--show-graphs]
```

Where

- `BASELINE` is either `rotation` or `jigsaw`,
- `MODEL` is one of
    - `resnet20`,
    - `resnet32`,
    - `resnet44`,
    - `resnet56`,
    - `resnet110`,
    - `resnet1202`,
    - `repvgg`,
    - `mobilenetv2`,
    - `densenet121`,
    - `densenet161`,
    - `densenet169`,
    - `shufflenet`,
    - `inceptionv3`,
    - `linear`,
    - `alexnet`,
    - `lenet5`,
    - `obc`
- `--show-graphs` will display the graphs of classification accuracy vs `BASELINE` accuracy for model `MODEL` on both
  interior and exterior datasets.

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

### Interior Domain

#### Rotation

| model       | R^2 (Interior Domain LR) | RMSE (Interior Domain LR) |
|-------------|--------------------------|---------------------------|
| densenet121 | 0.9759                   | 3.0703                    |
| resnet32    | 0.9635                   | 2.8954                    |
| resnet56    | 0.9592                   | 3.0014                    |
| resnet44    | 0.9577                   | 3.1458                    |
| resnet1202  | 0.9556                   | 2.7753                    |
| densenet161 | 0.9509                   | 4.4646                    |
| shufflenet  | 0.9479                   | 3.7456                    |
| resnet110   | 0.9402                   | 4.174                     |
| repvgg      | 0.9344                   | 5.2548                    |
| densenet169 | 0.9338                   | 5.0891                    |
| mobilenetv2 | 0.9269                   | 4.6029                    |
| resnet20    | 0.9218                   | 4.4542                    |
| inceptionv3 | 0.8202                   | 8.4792                    |
| linear      | 0.7647                   | 2.9008                    |
| alexnet     | 0.761                    | 4.2106                    |
| lenet5      | 0.7192                   | 4.4566                    |
| obc         | 0                        | 1.9596                    |

#### Jigsaw

| model       | R^2 (Interior Domain LR) | RMSE (Interior Domain LR) |
|-------------|--------------------------|---------------------------|
| resnet110   | 0.9169                   | 4.9211                    |
| resnet1202  | 0.8918                   | 4.3332                    |
| resnet44    | 0.8916                   | 5.0361                    |
| repvgg      | 0.891                    | 6.7776                    |
| densenet169 | 0.8721                   | 7.0742                    |
| densenet161 | 0.8652                   | 7.398                     |
| resnet56    | 0.8486                   | 5.7829                    |
| resnet32    | 0.8284                   | 6.2761                    |
| densenet121 | 0.823                    | 8.3171                    |
| shufflenet  | 0.8177                   | 7.0041                    |
| inceptionv3 | 0.8069                   | 8.7868                    |
| mobilenetv2 | 0.8012                   | 7.5913                    |
| lenet5      | 0.7983                   | 3.777                     |
| alexnet     | 0.7657                   | 4.1693                    |
| resnet20    | 0.7531                   | 7.9134                    |
| linear      | 0.6167                   | 3.7024                    |
| obc         | 0.0002                   | 1.9594                    |

### Exterior Domain

#### Rotation

| model       | R^2 (Exterior Domain LR) | RMSE (Exterior Domain LR) |
|-------------|--------------------------|---------------------------|
| resnet44    | 0.7787                   | 5.6932                    |
| densenet121 | 0.7574                   | 4.8347                    |
| alexnet     | 0.7455                   | 2.8506                    |
| resnet32    | 0.741                    | 5.8803                    |
| mobilenetv2 | 0.7337                   | 6.884                     |
| densenet169 | 0.7259                   | 5.4075                    |
| shufflenet  | 0.6967                   | 5.9666                    |
| repvgg      | 0.6488                   | 7.3129                    |
| resnet56    | 0.6467                   | 6.6604                    |
| resnet1202  | 0.6368                   | 6.4933                    |
| resnet110   | 0.6265                   | 5.6298                    |
| densenet161 | 0.5908                   | 5.5834                    |
| resnet20    | 0.4342                   | 8.4589                    |
| inceptionv3 | 0.3473                   | 8.6571                    |
| linear      | 0.2157                   | 3.3299                    |
| lenet5      | 0.0042                   | 4.922                     |
| obc         | 0                        | 1.8492                    |

#### Jigsaw

| model       | R^2 (Exterior Domain LR) | RMSE (Exterior Domain LR) |
|-------------|--------------------------|---------------------------|
| mobilenetv2 | 0.5254                   | 9.1901                    |
| resnet32    | 0.4956                   | 8.2067                    |
| resnet56    | 0.4314                   | 8.4495                    |
| linear      | 0.399                    | 2.9149                    |
| resnet44    | 0.3857                   | 9.4856                    |
| resnet110   | 0.3651                   | 7.3405                    |
| resnet1202  | 0.317                    | 8.9047                    |
| repvgg      | 0.2626                   | 10.596                    |
| densenet161 | 0.2247                   | 7.6857                    |
| alexnet     | 0.1634                   | 5.1686                    |
| densenet169 | 0.1559                   | 9.4892                    |
| resnet20    | 0.0815                   | 10.7777                   |
| shufflenet  | 0.0756                   | 10.4172                   |
| lenet5      | 0.0457                   | 4.8185                    |
| densenet121 | 0.0372                   | 9.6315                    |
| inceptionv3 | 0.0327                   | 10.5383                   |
| obc         | 0.0119                   | 1.8381                    |

### Exterior Domain with Interior Domain Fit

#### Rotation

| model       | R^2 (Exterior Domain w/ Interior Domain LR) | RMSE (Exterior Domain w/ Interior Domain LR) |
|-------------|---------------------------------------------|----------------------------------------------|
| alexnet     | 0.7266                                      | 2.9545                                       |
| resnet32    | 0.706                                       | 6.2656                                       |
| resnet44    | 0.7039                                      | 6.585                                        |
| repvgg      | 0.6295                                      | 7.5112                                       |
| densenet121 | 0.5837                                      | 6.3333                                       |
| mobilenetv2 | 0.519                                       | 9.252                                        |
| resnet1202  | 0.4952                                      | 7.6551                                       |
| resnet56    | 0.4449                                      | 8.3488                                       |
| resnet20    | 0.4248                                      | 8.5287                                       |
| densenet161 | 0.3994                                      | 6.7649                                       |
| shufflenet  | 0.2967                                      | 9.0862                                       |
| inceptionv3 | 0.1754                                      | 9.7304                                       |
| densenet169 | 0.0621                                      | 10.0026                                      |
| resnet110   | -0.206                                      | 10.1167                                      |
| lenet5      | -0.481                                      | 6.0026                                       |
| linear      | -1.3115                                     | 5.7166                                       |
| obc         | -5.1408                                     | 4.5824                                       |

#### Jigsaw

| model       | R^2 (Exterior Domain w/ Interior Domain LR) | RMSE (Exterior Domain w/ Interior Domain LR) |
|-------------|---------------------------------------------|----------------------------------------------|
| mobilenetv2 | 0.4531                                      | 9.8652                                       |
| resnet32    | 0.4454                                      | 8.6054                                       |
| resnet56    | 0.3434                                      | 9.0799                                       |
| resnet44    | 0.1312                                      | 11.2802                                      |
| alexnet     | -0.0896                                     | 5.8986                                       |
| repvgg      | -0.1504                                     | 13.235                                       |
| lenet5      | -0.2674                                     | 5.5529                                       |
| shufflenet  | -0.3547                                     | 12.6106                                      |
| resnet110   | -0.4204                                     | 10.9792                                      |
| resnet20    | -0.4587                                     | 13.5818                                      |
| resnet1202  | -0.5145                                     | 13.2599                                      |
| densenet169 | -0.8268                                     | 13.9598                                      |
| densenet121 | -1.3413                                     | 15.0196                                      |
| inceptionv3 | -2.2331                                     | 19.2668                                      |
| linear      | -2.2482                                     | 6.7766                                       |
| densenet161 | -3.0118                                     | 17.4833                                      |
| obc         | -5.138                                      | 4.5813                                       |
