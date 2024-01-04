Explain where to get the weights

## To do: add the weights to the google drive link
You can load the weights for the VGG-13bn model from this [Google Drive Link](https://drive.google.com/drive/folders/1aIWc87WfqGGtjGPHJe62tXzO6e5VoWcy?usp=sharing) and save it at `state_dicts/`

These weights and the code for the model were obtained from this [Github repository](https://github.com/huyvnphan/PyTorch_CIFAR10)


Run: conda create -n "env_name" python=3.9.18

package versions:

Python             3.9.18
matplotlib         3.8.2
numpy              1.26.2
torch              2.1.1
torchvision        0.16.1
progressbar        2.5
transformers       4.35.2

May have to change the load() function in ./site-package/torch/serialization.py to have MAP_LOCATION match yur device. I had to change it to 'cpu'. Shouldn't be a problem if using gpu.

Dowload the VGG weights from the drive and store them in VGG/vgg_models/state_dicts/

When using MLP, need to set wrokign directory like so: import os
os.chdir("/Users/charleslego/my_documents/ETH/Classes/Sem3/Deep_learning/Project/Code_Project/DLProject-MLP/Experiments/Charles/MLP/")

Don't forget to change the working directory in the hyper_search.py file




## Activation Maximization:

You can play around with the Act_Max_VGG_13.ipynb and Act_Max_B_12_W_1024.ipynb notebooks. By tuning hyperparameters and using different regularization techniques, you can get different results.

# Specific to VGG

You can run a hyperparameter search by running the hyper_search.py file. Don't forget to set your working directory manually at the top of the file.
To visualize the results from this hyperparameter search, you can run the visualization.ipynb notebook. All the tensors created by the hyper_search.py script will get saved as images in a folder called Act_Max_Img_MLP_Raw_Images. All these images are also available [here](https://drive.google.com/drive/u/3/folders/1FUrYC6vDdn8mwtCxXlNihVu6dKQtxJk3)

## Fast Gradient Sign Attack (FGSA):

With the fgsa_CNN_CIFAR10.py file you can create the adversarial datasets for each CNN model. You have to manually set your working directory, the model you want to use and the name of the .pth file you want to save the dataset data in.
With the fgsa_MLP_CIFAR10.py file you can create the adversarial datasets for the MLP model.

You can also download the datasets from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16mf4ZqYUmD8vvn82w1l78DJBiVik75gQ) and store them in a folder called cifar10_datasets

With the CIFAR10_CNN.ipynb notebook, you can evaluate the accuracy of the CNNs on all the adversarial datasets as well as the original CIFAR10 dataset. You have to manually choose which dataset you want to evaluate your model on and which model you want to use.
With the CIFAR10_MLP.ipynb notebook, you can do the same thing but with the MLP model.
