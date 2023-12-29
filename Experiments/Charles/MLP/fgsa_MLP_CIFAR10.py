import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from models.networks import get_model
from data_utils.data_stats import *
import torch.nn.functional as F
import torch.nn as nn

# Set the working directory
import os
os.chdir("/Users/charleslego/my_documents/ETH/Classes/Sem3/Deep_learning/Project/Code_Project/DLProject-MLP/Experiments/Charles/MLP/")


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Function that alters the image
def fgsa(image, label):
    eps_pga = 10   # Perturbation size

    # Create a copy of the image, but with gradient activated
    img_with_grad = resize(image.clone().detach().type(torch.float32).unsqueeze(0)).requires_grad_(True)
    
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

    return img_adv_sign.reshape(3,64,64)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load the CIFAR-10 test dataset -------------

resize = transforms.Resize(64)
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)


# ------------- Load the pretrained model ------------
dataset = 'cifar10'                 # One of cifar10, cifar100, stl10, imagenet or imagenet21
architecture = 'B_12-Wi_1024'
data_resolution = 32                # Resolution of data as it is stored
crop_resolution = 64                # Resolution of fine-tuned model (64 for all models we provide)
num_classes = CLASS_DICT[dataset]
data_path = './beton/'
eval_batch_size = 1024
checkpoint = 'in21k_cifar10'        # This means you want the network pre-trained on ImageNet21k and finetuned on CIFAR10

model = get_model(architecture=architecture, resolution=crop_resolution, num_classes=CLASS_DICT[dataset],
                  checkpoint='in21k_cifar10')
model = nn.Sequential(
    nn.Flatten(1, -1),model
)
model = model.to(device)
model.train()

# Create a file that can used used to load a modified dataset

modified_images = []
labels = []

for i in range(len(test_dataset)):
    image, label = test_dataset[i]

    transform = transforms.Compose([transforms.PILToTensor()])
    image = transform(image) 
    label = torch.tensor(label)

    # Get FGSA adversarial example
    modified_image = fgsa(image, label)
    modified_images.append(Image.fromarray(modified_image.detach().permute(1,2,0).numpy().astype(np.uint8)))
    labels.append(label)

# Apply the transform to each image and store in a dictionary
dataset_dict = [{'image': img, 'label': label} for img, label in zip(modified_images, labels)]

# Save the dataset dictionary to a file
torch.save(dataset_dict, f'cifar10_fgsa_MLP.pth')