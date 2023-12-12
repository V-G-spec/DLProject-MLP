# DLProject- MLP

## Installation
### Data
You can download the `train_32.beton` file from this [Google Drive Link](https://drive.google.com/drive/folders/16114hZHtzcx3UXa2FMGlNGh-jTjWB-cz?usp=drive_link) and save it at `Data/`
### [CHECK] Environment
Set up a conda environment using the `environment.yml` file as follows:
```
conda env create -f environment.yml
conda activate interpret_MLP
```

### [WIP] Environment pip
#### Create and activate a virtual environment as follows:
##### For Windows:
```
virtualenv interpret_MLP
interpret_MLP\Scripts\activate
```
##### For MacOS and Linux:
```
virtualenv interpret_MLP
source interpret_MLP/bin/activate
```
#### Install the required packages:
```
pip install -r requirements.txt
```

## Proposal
### 1. Introduction
An instance of deep neural networks that is frequently utilized to analyze visual imagery is the convolutional neural network (CNN) (LeCun et al., 1989). CNNs feature built-in assumptions (“inductive biases” (Cao & Wu, 2021)), such as shift-equivariances, about the nature of computer vision tasks, which make them particularly suited to such problems. However, theoretical work in deep learning largely focuses on the much simpler multi-layer perceptron (MLP) (Sanger & Baljekar, 1958). This motivated Bachmann et al. (2023) to test MLPs on vision tasks. Intriguingly, they discovered that when given sufficient computational resources, large MLP models can exhibit strong performance as image classifiers despite their lack of inductive bias. This work aims to investigate whether MLPs applied to vision tasks approximate their CNN counterparts. To test this hypothesis, we will compare a CNN and an MLP using two methods: Activation Maximization (Section 2) and Dictionary Learning (Section 3).
### 2. Activation Maximization
Activation Maximization (AM) involves optimising the activation of a given neuron or layer of a network by performing gradient ascent on the input (Erhan et al., 2009). It has been successfully applied to many CNNs (Olah et al., 2017; Øygard, 2015; Nguyen et al., 2019). As MLPs are not a common tool in vision tasks, there has been very little effort to try to apply AM to MLPs. We are interested in investigating how images produced using AM on MLPs compare with those obtained from different architectures such as CNNs. More specifically, many visualizations extracted from CNNs contain recurring patterns (Olah et al., 2017), which we hypothesize emerge from the innate shift-invariance property of CNNs. If MLPs indeed approximate the inductive biases in CNNs, they should generate similar recurring patterns after AM (Prediction 1).
### 3. Autoencoder for Dictionary Learning
Individual neurons in deep neural networks respond to a wide variety of input features (Bricken et al., 2023). This results in dense hidden activations and ultimately “entangled” representations. According to the superposition hypothesis (Elhage et al., 2022), the vector representing the activations of a hidden MLP layer in response to a particular input can be expressed as a sparse linear combination of an overcomplete set of non-orthogonal basis directions. In this alternate high-dimensional basis, the activation vectors are sparse, resulting in less “entangled” representations that are easier to interpret (Bricken et al., 2023). In (Bricken et al., 2023), it has been shown that an autoencoder with one hidden layer can be used to find such a basis (referred to as a dictionary (Zhang et al., 2019)). Importantly, the authors show that the autoencoder learns relatively similar features (i.e., basis directions) for different models trained on the same data, which makes this method particularly suited for comparing MLPs with CNNs. We propose to train an autoencoder with the same architecture as in (Bricken et al., 2023) on the activations in the last hidden layer of both a CNN and an MLP to test whether the autoencoder learns similar features (Prediction 2) in both cases. We quantify feature similarity as the correlation between feature activations.
### 4. Models and Dataset
We compare two pre-trained models: “VGG13 bn”, a VGG network (Simonyan & Zisserman, 2015) with 13 convolutional layers (≈ 28M parameters, available at (Huyvnphan, 2023)) and “B-12/Wi-1024”, an MLP model with 12 blocks and layer width 1024 (≈ 124M parameters, available at (Bachmann et al., 2023)). We will focus on the CIFAR-10 dataset (Krizhevsky & Hinton, 2009) because both models demonstrate strong performance on it, and the relatively small number of output classes simplifies interpretation. For the Dictionary Learning method, we will train an autoencoder with one hidden layer featuring three times as many neurons as the input and output layers (overall ≈ 100M/6M parameters when trained on CNN/MLP). We have experimentally verified the computational feasibility of both of our methods when applied to these two models.
### 5. Evaluation
Since this is a fairly exploratory project, there is no easy way to quantify success. Rather, we see this as a sort of hypothesis test which we consider successful if we produce evidence in support of or against Predictions 1 and 2.
