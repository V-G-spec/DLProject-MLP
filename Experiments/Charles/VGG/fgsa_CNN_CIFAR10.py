import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from models.vgg import vgg13_bn
from models.densenet import densenet169
from models.resnet import resnet50
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set the working directory, change to your specific directory
import os
os.chdir("/Users/charleslego/my_documents/ETH/Classes/Sem3/Deep_learning/Project/Code_Project/DLProject-MLP/Experiments/Charles/VGG/")


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
model = densenet169(pretrained=True) # choose from the different models: vgg13_bn, densenet169, resnet50
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

    # Get FGSA adversarial example
    modified_image = fgsa(image, label)
    modified_images.append(Image.fromarray(modified_image.detach().permute(1,2,0).numpy().astype(np.uint8)))
    labels.append(label)

# Apply the transform to each image and store in a dictionary
dataset_dict = [{'image': img, 'label': label} for img, label in zip(modified_images, labels)]

# Save the dataset dictionary to a file
torch.save(dataset_dict, f'cifar10_fgsa_densenet169.pth') # Change name accordingly