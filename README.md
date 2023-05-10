# DataCV Challenge @ CVPR 2023

Welcome to DataCV Challenge 2023!

This is the development kit repository for [the 1st DataCV Challenge](https://sites.google.com/view/vdu-cvpr23/competition?authuser=0). This repository includes details on how to download datasets, run baseline models, and organize your result as `answer.zip`. The final evaluation will occur on the [CodeLab evaluation server](https://codalab.lisn.upsaclay.fr/competitions/10221), where all competition information, rules, and dates can be found.

More specifically, this is a fork of said repository for the research paper titled **Is the Linear Correlation Between Classification Accuracy and Self-Supervision Accuracy Preserved Between Model Architectures?** We have amended each section to explain our problem.
## Table of Contents

- [DataCV Challenge @ CVPR 2023](#datacv-challenge--cvpr-2023)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Related Work](#related-work)
  - [Challenge Data](#challenge-data)
  - [Classifiers being Evaluated](#classifiers-being-evaluated)
  - [Organize Results for Submission](#organize-results-for-submission)
  - [Several Baselines](#several-baselines)
  	- [Baseline Description](#baseline-description)
    - [Baseline Results](#baseline-results)
        - [ResNet-56](#resnet-56)
        - [RepVGG-A0](#repvgg-a0)
    
  - [Code-Execution](#code-execution)
  
## Abstract
We verify that the linear correlation between classification accuracy and self-supervision accuracy for rotation prediction and jigsaw classification holds between various computer vision neural network models over the CIFAR-10 dataset. We observed that there is a medium to strong linear correlation between classification accuracy and rotation prediction accuracy, while the correlation between classification accuracy and jigsaw classification accuracy is much weaker, to the extent that some models do not exhibit any correlation at all. As a consequence, it is not always possible to accurately judge the performance of a model on a supervised task given the same model's performance on a self-supervised task.
## Overview
Constructing a dataset of images for a supervised task such as classification requires generating labels for every image.
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
Aside from the work that this repository was forked from [ref], our work also focuses on the findings of [ref] which observed a strong linear correlation between a model's performance on image classification and rotation prediction over MNIST, CIFAR10 and ImageNet, where for each dataset, different models were used. Our work keeps the dataset (i.e: CIFAR-10) fixed, and we test various models on this dataset to verify that the linear correlation between classification accuracy and rotation prediction accuracy is preserved. In the future work section of the same paper, the writers also remark that there is a strong linear correlation between classification accuracy and jigsaw classification accuracy. We verify this finding in our work as well.
#### CIFAR-10 Dataset
#### ResNet
#### DenseNet
#### MobileNetv2
#### ShuffleNet
#### RepVGG
#### Image Classification
#### Self-Supervised Tasks

## Datasets Used
The training dataset consists of 1,000 transformed datasets from the original [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test set, using the transformation strategy proposed by [Deng et al. (2021)](https://arxiv.org/abs/2007.02915). The validation set was composed of [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1), [CIFAR-10.1-C](https://github.com/hendrycks/robustness) (add corruptions [(Hendrycks et al., 2019)](https://arxiv.org/abs/1903.12261) to [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) dataset), and CIFAR-10-F (real-world images collected from [Flickr](https://www.flickr.com))

The CIFAR-10.1 dataset is a single dataset. In contrast, CIFAR-10.1-C and CIFAR-10-F contain 19 and 20 datasets, respectively. Therefore, the total number of datasets in the validation set is 40.

The training datasets share a common label file named `labels.npy`, and images files are named `new_data_xxx.npy`, where `xxx` is a number from 000 to 999. For every dataset in the validation set, the image file and their labels are stored as two separate `Numpy` array files named "data.npy" and "labels.npy". The PyTorch implementation of the Dataset class for loading the data can be found in `utils.py`.

Download the training datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Eb9yO_Qg41lOkoRS7P6gmqMBk5Q6A2gCV8YbRbuLpB8NwQ?e=WO3Gqi)

Download the validation datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Edg83yRxM9BPonPP22suB_IBrHlKYV5bOn4VK-c5RZ8dtQ?e=kExXEm)

Download the training datasets' accuracies on the ResNet-56 model: [link](https://anu365-my.sharepoint.com/:t:/g/personal/u7136359_anu_edu_au/EQ4XcZLeVPNAg45JdB0mZ4ABO6nsIDDD3z2_frx0rnbRpg?e=5wA3Xi)

Download the training datasets' accuracies on the RepVGG-A0 model: [link](https://anu365-my.sharepoint.com/:t:/g/personal/u7136359_anu_edu_au/EWUVPpAqYcNJq8iB4AfYD7oBodhHMI1B_1Mijd7x8V8xlA?e=oPDaL3)

**NOTE: To access the test datasets and participate in the competition, please fill in the [Datasets Request Form](https://anu365-my.sharepoint.com/:b:/g/personal/u7136359_anu_edu_au/ERz4ANQ1A31PvJKgd3mNxr8B1F4e0zfaZL3P_NLOvKrivg?e=lG7mkL) and send the signed form to [the competition organiser](mailto:datacvchallenge2023@gmail.com;VDU2023@gmail.com). Failing to provide the form will lead to the revocation of the CodaLab account in the competition.**

## Classifiers being Evaluated
In this competition, the classifiers being evaluated are ResNet-56 and RepVGG-A0. Both implementations can be accessed in the public repository at https://github.com/chenyaofo/pytorch-cifar-models. To utilize the models and load their pretrained weights, use the code provided.
```python
import torch

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56"， pretrained=True)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
```

## Organize Results for Submission
As we use automated evaluation scripts for submissions, the format of submitted files is very important. So, there is a function to store the accuracy predictions into the required format, named `store_ans` in `code/utils.py` and `results_format/get_answer_txt.py`.

Read the `results_format/get_answer_txt.py` file to comprehend the function usage. Execute the code below to see the results.

```bash
python3 results_format/get_answer_txt.py
```

## Several Baselines
The necessary dependencies are specified in the `requirements.txt` file and the experiments were conducted using Python version 3.10.8, with a single GeForce RTX 2080 Ti GPU.

The table presented below displays the results of the foundational measurements using [root-mean-square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE). Accuracies are converted into percentages prior to calculation.

### Baseline Results
#### ResNet-56

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 7.285  | 6.386  | 7.763  | 7.129  |
| ConfScore | 2.190  | 9.743  | 2.676  | 6.985  |
| Entropy   | 2.424  | 10.300 | 2.913  | 7.402  |
| ATC       | 11.428 | 5.964  | 8.960  | 7.766  |
| FID       | 7.517  | 5.145  | 4.662  | 4.985  |

#### RepVGG-A0

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 16.726 | 17.137 | 8.105 | 13.391 |
| ConfScore | 5.470  | 12.004 | 3.709 | 8.722  |
| Entropy   | 5.997  | 12.645 | 3.419 | 9.093  |
| ATC       | 15.168 | 8.050  | 7.694 | 8.132  |
| FID       | 10.718 | 6.318  | 5.245 | 5.966  |

### Code-Execution/Method
To install required Python libraries, execute the code below.
```bash
pip3 install -r requirements.txt
```
The above results can be replicated by executing the code provided below in the terminal.
```bash
cd code/
chmod u+x run_baselines.sh && ./run_baselines.sh
```
To run one specific baseline, use the code below.
```bash
cd code/
python3 get_accuracy.py --model <resnet/repvgg> --dataset_path DATASET_PATH
python3 baselines/BASELINE.py --model <resnet/repvgg> --dataset_path DATASET_PATH
```

###  Baseline Description
The following succinctly outlines the methodology of each method, as detailed in the appendix of "Predicting Out-of-Distribution Error with the Projection Norm" paper [(Yu et al., 2022)](https://arxiv.org/abs/2202.05834).

**Rotation.** The *Rotation Prediction* (Rotation) [(Deng et al., 2021)](https://arxiv.org/abs/2106.05961) metric is defined as
```math
\text{Rotation} = \frac{1}{m}\sum_{j=1}^m\Big\{\frac{1}{4}\sum_{r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}}\mathbf{1}\{C^r(\tilde{x}_j; \hat{\theta}) \neq y_r\}\Big\},
```
where $y_r$ is the label for $r \in \lbrace 0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ} \rbrace$, and $C^r(\tilde{x}_j; \hat{\theta})$ predicts the rotation degree of an image $\tilde{x}_j$.

**ConfScore.** The *Averaged Confidence* (ConfScore) [(Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1610.02136) is defined as
$$\text{ConfScore} = \frac{1}{m}\sum_{j=1}^m \max_{k} \text{Softmax}(f(\tilde{x}_j; \hat{\theta}))_k,$$
where $\text{Softmax}(\cdot)$ is the softmax function.

**Entropy.** The *Entropy* [(Guillory et al., 2021)](https://arxiv.org/abs/2107.03315) metric is defined as
```math
\text{Entropy} = \frac{1}{m}\sum_{j=1}^m \text{Ent}(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))),$$
where $\text{Ent}(p)=-\sum^K_{k=1}p_k\cdot\log(p_k)$.
```

**ATC.** The *Averaged Threshold Confidence* (ATC) [(Garg et al., 2022)](https://arxiv.org/abs/2201.04234) is defined as
```math
\text{ATC} = \frac{1}{m}\sum_{j=1}^m\mathbf{1}\{s(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))) < t\},$$
where $s(p)=\sum^K_{j=1}p_k\log(p_k)$, and $t$ is defined as the solution to the following equation,
$$\frac{1}{m^{\text{val}}} \sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{s(\text{Softmax}(f(x_{\ell}^{\text{val}}; \hat{\theta}))) < t\} = \frac{1}{m^{\text{val}}}\sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{C(x_\ell^{\text{val}}; \hat{\theta}) \neq y_\ell^{\text{val}}\}
```
where $(x_\ell^{\text{val}}, y_\ell^{\text{val}}), \ell=1,\dots, m^{\text{val}}$, are in-distribution validation samples.

**FID.** The *Frechet Distance* (FD) between datasets [(Deng et al., 2020)](https://arxiv.org/abs/2007.02915) is defined as
```math
\text{FD}(\mathcal{D}_{ori}, \mathcal{D}) = \lvert \lvert \mu_{ori} - \mu \rvert \rvert_2^2 + Tr(\Sigma_{ori} + \Sigma - 2(\Sigma_{ori}\Sigma)^\frac{1}{2}),
```
where $\mu\_{ori}$ and $\mu$ are the mean feature vectors of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. $\Sigma\_{ori}$ and $\Sigma$ are the covariance matrices of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. They are calculated from the image features in $\mathcal{D}\_{ori}$ and $\mathcal{D}$, which are extracted using the classifier $f\_{\theta}$ trained on $\mathcal{D}\_{ori}$.

The Frechet Distance calculation functions utilized in this analysis were sourced from a [publicly available repository](https://github.com/Simon4Yan/Meta-set) by [Weijian Deng](https://github.com/Simon4Yan).
