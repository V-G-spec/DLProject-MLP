# Set the working directory, change to your specific directory
project_dir = '/Location_of_src_folder'

import sys
import os

sys.path.insert(0,project_dir)
os.chdir(project_dir)

import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from CNN_models.vgg import vgg13_bn
from CNN_models.densenet import densenet169
from CNN_models.resnet import resnet50
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ------------ Fast Gradient Sign Attack function ---------------
def fgsa(image, label):
    eps_pga = 10   # Perturbation size

    # Create a copy of the image, but with gradient activated
    img_with_grad = image.clone().detach().type(torch.float32).unsqueeze(0).requires_grad_(True)

    # Calculate logits
    logits_sign = model(normalize(img_with_grad))
    
    # Calculate the loss
    loss_sign = F.cross_entropy(logits_sign, label.unsqueeze(0))
    # Calculate the loss w.r.t to the input
    loss_sign.backward()
    # Create adversarial perturbation 'in one step' with sign of the gradient
    r_sign = eps_pga * torch.sign(img_with_grad.grad.data)
    # Create the adversarial example
    img_adv_sign = img_with_grad + r_sign

    return img_adv_sign.reshape(3,32,32)


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load the CIFAR-10 test dataset -------------

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

# ------------- Load the pretrained model ------------
models = [densenet169(pretrained=True), resnet50(pretrained=True), vgg13_bn(pretrained=True)]
models_name = ['densenet169', 'resnet50', 'vgg13_bn']
for model, model_name in zip(models, models_name):
    model = model.to(device)
    model.eval()

    # Create a file that can used used to load a modified dataset

    modified_images = []
    labels = []

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        transform = transforms.Compose([transforms.PILToTensor()])

        image = transform(image) 
        label = torch.tensor(label)
        image, label = image.to(device), label.to(device)

        # Get FGSA adversarial example
        modified_image = fgsa(image, label)
        modified_images.append(Image.fromarray(modified_image.detach().permute(1,2,0).cpu().numpy().astype(np.uint8)))
        labels.append(label)

    # Apply the transform to each image and store in a dictionary
    dataset_dict = [{'image': img, 'label': label} for img, label in zip(modified_images, labels)]

    # Save the dataset dictionary to a file
    torch.save(dataset_dict, f'cifar10_FGSM_{model_name}.pth') # Change name accordingly
