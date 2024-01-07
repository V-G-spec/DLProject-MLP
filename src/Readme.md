# DLProject- MLP

## General information
The pretrained CNN models were obtained from this [Github repository](https://github.com/huyvnphan/PyTorch_CIFAR10)
The pretrained MLP model was obtained from this [Github repository](https://github.com/gregorbachmann/scaling_mlps)

You can download the weights for the three CNN models: VGG-13bn, ResNet50 and DenseNet169 from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16114hZHtzcx3UXa2FMGlNGh-jTjWB-cz) and save them under `CNN_models/state_dicts/`

#### Create and activate a virtual environment as follows:
##### For Windows:
```
python -m venv ./interpret_MLP
interpret_MLP\Scripts\activate
```
##### For MacOS and Linux:
```
python -m venv ./interpret_MLP
source interpret_MLP/bin/activate
```
#### Install the required packages:
```
pip install -r requirements.txt
```

## Act_Max_&_FGSA folder
In the Act_Max_&_FGSA folder, you have all the files necessary to reproduce the results of "Activation maximization" and "Adversarial attacks", Sections 3.1 and 3.2 of our project.

### Activation Maximization:
In all files you will have to set the working directory at the top of the file.

You can play around with the Act_Max_VGG_13.ipynb and Act_Max_B_12_W_1024.ipynb notebooks. By tuning hyperparameters and using different regularization techniques, you can get different results.

If running the Act_Max_B_12_W_1024.ipynb on a machine with cpu only, you will have to change the load() function in your_env/lib/python3.*/site-package/torch/serialization.py  to have map_location: MAP_LOCATION = 'cpu'
Then restart the kernel and import all the necessary packages again.
This is not necessary if your machine has a gpu.

#### Specific to VGG

You can run a hyperparameter search by running the hyper_search.py file. Don't forget to set your working directory manually at the top of the file.
To visualize the results from this hyperparameter search, you can run the visualization.ipynb notebook. All the tensors created by the hyper_search.py script will get saved as images in a folder called Act_Max_Img_MLP_Raw_Images. All these images are also available [here](https://drive.google.com/drive/u/3/folders/1FUrYC6vDdn8mwtCxXlNihVu6dKQtxJk3)

### Fast Gradient Sign Attack (FGSA):

With the fgsa_CNN_CIFAR10.py file you can create the adversarial datasets for each CNN model. You have to manually set the model you want to use and the name of the .pth file you want to save the dataset data in.
With the fgsa_MLP_CIFAR10.py file you can create the adversarial datasets for the MLP model.
Once you have generated the datasets, store them under `Act_Max_&_FGSA/cifar10_datasets/`

You can also download the datasets from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16mf4ZqYUmD8vvn82w1l78DJBiVik75gQ) and store them under `Act_Max_&_FGSA/cifar10_datasets/`

With the CIFAR10_CNN.ipynb notebook, you can evaluate the accuracy of the CNNs on all the adversarial datasets as well as the original CIFAR10 dataset. You have to manually choose which dataset you want to evaluate your model on and which model you want to use.
With the CIFAR10_MLP.ipynb notebook, you can do the same thing but with the MLP model.

## dictionary_learning folder

The folder dictionary_learning contains the scripts to reproduce the "Dictionary Learning with Sparse Autoencoders" section (2.4) of our project.

As a first step, you can collect last-layer activations for the MLP and the VGG by running the notebooks harvest_activations_mlp.ipynb and harvest_activations_vgg.ipynb, respectively. If you don't want to do that you can simply download the activations from the Google Drive at the bottom of this ReadMe ( acts_B_12-Wi_1024_cifar10_test_postskip.h5 and acts_VGG13_bn_cifar10_test.h5, respectively). Perhaps confusingly, the sparse autoencoders (SAEs) are trained on activations elicited by images from the CIFAR10 ***test*** set.

Next run train_autoencoder_for_mlp.py and train_autoencoder_for_mlp to train the sparse autoencoders on the collected activations. You will have to change the location of the activations file at line 61. If you prefer not to train the models yourself, you can just download the model parameters from the Google Drive at the bottom of this ReadMe (SAE_100_epochs_bs_32_CIFAR10_test_B_12-Wi_1024_postskip.pt and SAE_100_epochs_bs_32_CIFAR10_test_vgg_bn13.pt, respectively).

Now, download the images from the CIFAR10 ***training*** set, together with their targets from the Google Drive at the bottom of this ReadMe (targs_cifar10_train.h5 and ims_cifar10_train.h5). This is important to resproduce our feature analysis results because the images are in the same order as when we first did the analysis.

Once you have set the working directroy at the top of the files, you are all set to run the feature analysis notebooks feature_analysis_mlp.ipynb and feature_analysis_vgg.ipynb.

All the data loaded by the scripts can be found here: [Google Drive Link](https://drive.google.com/drive/folders/1LVX-Qd6mpycTucePQl6INerHgdfTzkPc?usp=sharing). Place all files you download in the folder this ReadMe is located in.
