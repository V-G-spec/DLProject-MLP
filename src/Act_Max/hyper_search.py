# Set the working directory, change to your specific directory
project_dir = '/Your_project_folder'

import sys
import os

sys.path.insert(0,project_dir)
os.chdir(project_dir)

import torch
from models.networks import get_model
from torch import nn
from torchvision import transforms
from data_utils.data_stats import *
import numpy as np

# Set seed values for PyTorch and NumPy for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

def main():

    # ---------Defining important parameters for loading the model-------------
    dataset = 'cifar10'                 # One of cifar10, cifar100, stl10, imagenet or imagenet21
    architecture = 'B_12-Wi_1024'
    data_resolution = 32                # Resolution of data as it is stored
    crop_resolution = 64                # Resolution of fine-tuned model (64 for all models we provide)
    num_classes = CLASS_DICT[dataset]
    data_path = './beton/'
    eval_batch_size = 1024
    checkpoint = 'in21k_cifar10'        # This means you want the network pre-trained on ImageNet21k and finetuned on CIFAR10

    # ----------Define the model and specify the pre-trained weights--------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(architecture=architecture, resolution=crop_resolution, num_classes=CLASS_DICT[dataset],
                    checkpoint='in21k_cifar10')
    model = nn.Sequential(nn.Flatten(1, -1),model)
    model.to(device)


    # ---------The different hyperparameters to tune-----------
    start_sss = np.logspace(-5,-1,5) # start step sizes
    kernel_sizes = [3,5,7]
    start_sigs = np.linspace(0.1,1,5)
    class_idxs = [i for i in range(10)]
    
    epochs = 1000
    num_im_save = 10


    # -------- Hyperparameter tuning loop---------
    for start_ss in start_sss:
        end_ss = start_ss/10
        for kernel_size in kernel_sizes:
            for start_sig in start_sigs:
                end_sig = start_sig/10
                for idx in class_idxs:

                    # ----------Creating a random starting input-----------
                    mean = 0.5
                    std = 0.5/3
                    input = mean + std*torch.randn((3,64,64)) # should be of size 3,64,64
                    input.unsqueeze_(0) # create a mini-batch as expected by the model
                    input = input.to(device)
                    input.requires_grad_()

                    for epoch in range(epochs):

                        model.zero_grad()

                        if input.grad != None:
                            input.grad.zero_()

                        y_pred = model.forward(input)
                        
                        y_pred[0,idx].backward()

                        g = input.grad
                        
                        with torch.no_grad():

                            step_size = start_ss + ((end_ss - start_ss) * epoch) / epochs
                            input += step_size/np.abs(g.cpu()).mean() * g

                            sig = start_sig + ((end_sig - start_sig) * epoch) / epochs
                            blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sig)
                            input = blurrer(input)    

                        input.requires_grad_()

                        if (epoch+1) % (epochs/num_im_save) == 0:
                            output = input.detach()
                            output_directory = './Act_Max_Img_MLP/'
                            output_filename = f'class_{idx}_ker_{kernel_size}_startss_{start_ss}_startsig_{start_sig}_epoch_{epoch+1}.pt'
                            output_path = output_directory + output_filename
                            torch.save(output, output_path)
                            



if __name__ == "__main__":
    main()