# Interpreting Vision Models: A Comparative Study of MLPs and CNNs

## Overview
This repository contains code and resources related to our course project for [Deep Learning](https://da.inf.ethz.ch/teaching/2023/DeepLearning/), offered in the Fall Semester 2023 at ETH Zurich. The pre-trained Convolutional Neural Network (CNN) models utilized in this project were obtained from the [PyTorch_CIFAR10 GitHub repository](https://github.com/huyvnphan/PyTorch_CIFAR10). Additionally, the pre-trained Multilayer Perceptron (MLP) model can be found in the [scaling_mlps GitHub repository](https://github.com/gregorbachmann/scaling_mlps).

The weights for the CNN models (VGG-13bn, ResNet50, and DenseNet169) are available for download from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16114hZHtzcx3UXa2FMGlNGh-jTjWB-cz). Please save these weights under `src/CNN_models/state_dicts/`.

## Environment Setup
To replicate our experiments, follow these steps to create and activate a virtual environment:

### For Windows:
```bash
python -m venv ./interpret_MLP
interpret_MLP\Scripts\activate
```

### For MacOS and Linux:
```bash
python -m venv ./interpret_MLP
source interpret_MLP/bin/activate
```

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

Note: We run our experiments on Python version 3.9.18 (as included in `requirements.txt`). However, we recommend users to use the correct python executable to initiate the virtual environment

## Activation Maximization
In the `src/Act_Max` folder, you will find all the files necessary to reproduce the "Activation Maximization" results corresponding to Section 3.1 of our project.

### Activation Maximization:
Adjust the working directory at the top of each file to point to the `/src` folder. Explore `Act_Max_VGG_13.ipynb` and `Act_Max_B_12_W_1024.ipynb` notebooks to experiment with hyperparameters and regularization techniques.

For running `Act_Max_B_12_W_1024.ipynb` on a CPU-only machine, modify the `load()` function in `your_env/lib/python3.*/site-package/torch/serialization.py` to have `map_location: MAP_LOCATION = 'cpu'`. Restart the kernel and import the necessary packages. This step is not necessary for GPU machines.

#### MLP
Conduct a hyperparameter search using `hyper_search.py`. Set the working directory manually. Visualize results with the `visualization.ipynb` notebook. Images created by `hyper_search.py` are available [here](https://drive.google.com/drive/u/3/folders/1FUrYC6vDdn8mwtCxXlNihVu6dKQtxJk3).

Produce the CIFAR-10 dataset with images transformed into randomly ordered tiles with `mixed_up_cifar_10.py`. Set the working directory manually. Store generated datasets under `src/cifar10_datasets/`.

Alternatively, download datasets from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16mf4ZqYUmD8vvn82w1l78DJBiVik75gQ) and save them under `src/cifar10_datasets/`.

You can evaluate the accuracy of the MLP and VGG models together with FGSM-perturbed dataset in the following section.

## Fast Gradient Sign Attack (FGSM):
In the `src/FGSM` folder, you will find all the files necessary to reproduce the results of "Adversarial Attacks" corresponding to Section 3.2 of our project.

Adjust the working directory at the top of each file. Use `FGSM_CNN_CIFAR10.py` and `FGSM_MLP_CIFAR10.py` to create adversarial datasets for each model. Manually set the model and file name. Store generated datasets under `src/cifar10_datasets/`.

Alternatively, download datasets from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16mf4ZqYUmD8vvn82w1l78DJBiVik75gQ) and save them under `src/cifar10_datasets/`.

Evaluate CNNs on adversarial datasets and the original CIFAR10 dataset using `CIFAR10_CNN.ipynb`. For MLP, use `CIFAR10_MLP.ipynb`.

## Dictionary Learning
The `src/Dictionary_learning` folder contains scripts to reproduce our project's "Dictionary Learning with Sparse Autoencoders" section (3.3).

1. Collect last-layer activations for MLP and VGG using `harvest_activations_mlp.ipynb` and `harvest_activations_vgg.ipynb`. Adjust the working directory at the top of each file.
   - Optionally, download activations from the Google Drive at the end of this ReadMe (`acts_B_12-Wi_1024_cifar10_test_postskip.h5` and `acts_VGG13_bn_cifar10_test.h5`, respectively). Perhaps confusingly, the sparse autoencoders (SAEs) are trained on activations elicited by images from the CIFAR10 ***test*** set.

2. Run `train_autoencoder_for_mlp.py` and `train_autoencoder_for_vgg.py` to train the sparse autoencoders on the collected activations. You will have to change the location of the activations file at line 61.
   - If you prefer not to train the models yourself, you can download the model parameters from the Google Drive at the bottom of this ReadMe (`SAE_100_epochs_bs_32_CIFAR10_test_B_12-Wi_1024_postskip.pt` and `SAE_100_epochs_bs_32_CIFAR10_test_vgg_bn13.pt`, respectively).

3. Download the images from the CIFAR10 ***training*** set, together with their targets from the Google Drive at the bottom of this ReadMe (`targs_cifar10_train.h5` and `ims_cifar10_train.h5`)
   - This is important to reproduce our feature analysis results because the images are in the same order as when we first did the analysis.

After setting the working directory in the files, run `feature_analysis_mlp.ipynb` and `feature_analysis_vgg.ipynb` to conduct feature analysis

All the data loaded by the scripts can be found here: [Google Drive Link](https://drive.google.com/drive/folders/1LVX-Qd6mpycTucePQl6INerHgdfTzkPc?usp=sharing). Place all files you download in the `dictionary_learning` folder.
