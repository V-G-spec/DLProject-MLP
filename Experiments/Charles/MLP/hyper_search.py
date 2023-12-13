import torch
from models.networks import get_model
from torch import nn
from torchvision import transforms
from data_utils.data_stats import *
import numpy as np


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
    model = get_model(architecture=architecture, resolution=crop_resolution, num_classes=CLASS_DICT[dataset],
                  checkpoint='in21k_cifar10')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model and specify the pre-trained weights
    model = get_model(architecture=architecture, resolution=crop_resolution, num_classes=CLASS_DICT[dataset],
                    checkpoint='in21k_cifar10')
    model = nn.Sequential(nn.Flatten(1, -1),model)
    model.to(device)


    # ----------Creating a random starting input-----------
    mean = 0.5
    std = 0.03
    input = mean + std*torch.randn((3,64,64)) # should be of size 3,64,64
    input.unsqueeze_(0) # create a mini-batch as expected by the model
    input = input.to(device)
    input.requires_grad_()

    # ---------The different hyperparameters to tune-----------
    num_epochs = []
    start_sss = [] # start step sizes
    end_sss = [] # end step sizes
    start_sigs = []
    end_sigs = []
    theta_decays = []
    class_idxs = []

    num_im_save = 10


    # -------- [TO DO] write the rest of the hyperparameter tuning loop---------
    # Class I want to optimize
    idx = 5 # 0: airplanes, 1: cars, 2: birds, 3: cats, 4: deer, 5: dogs, 6: frogs, 7: horses, 8: ships, 9: trucks


    epochs = 1000
    start_ss = 0.001
    end_ss = 0.001


    start_sig = 1 # Having a decaying sigma seems to yield better results
    end_sig = 0.5 
    theta_decay = 0.0001 # Theta decay seems to help (0.02 seems good)

    for epoch in range(epochs):

        model.zero_grad()

        if input.grad != None:
            input.grad.zero_()

        y_pred = model.forward(input)
        
        y_pred[0,idx].backward()

        g = input.grad
        
        with torch.no_grad():

            step_size = start_ss + ((end_ss - ss) * epoch) / epochs
            input += step_size/np.abs(g).mean() * g

            input = input.mul((1.0 - theta_decay)) # weight decay

            sig = start_sig + ((end_sig - start_sig) * epoch) / epochs
            blurrer = transforms.GaussianBlur(kernel_size=5, sigma=sig)
            input = blurrer(input)    

        input.requires_grad_()

        if (epoch+1) % (epochs/num_im_save) == 0:
            output = input.detach().numpy()
            output_directory = './Act_Max_Img/MLP/'
            output_filename = f'class_{idx}_slr_{start_ss}_elr_{end_ss}_ssig_{start_sig}_esig_{end_sig}_thetad_{theta_decay}_epoch_{epoch}.jpg'
            output_path = output_directory + output_filename
            np.savetxt(output_path, output, delimiter=",")



if __name__ == "__main__":
    main()