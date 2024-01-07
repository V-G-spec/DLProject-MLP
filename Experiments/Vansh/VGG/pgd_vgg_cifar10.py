import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from vgg_models.vgg import vgg13_bn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
epsil = 8/255
alph = 2/255

# Function that performs PGD attack
def pgda(image, label, epsilon=epsil, alpha=alph, num_steps=10):
    # Create a copy of the image, but with gradient activated
    image = image.to(device)
    label = label.to(device)
    img_with_grad = image.clone().detach().type(torch.float32).unsqueeze(0).requires_grad_(True).to(device)
    
    for _ in range(num_steps):
        # Calculate logits
        logits_pgd = model(normalize(img_with_grad))
        # Calculate the loss
        loss_pgd = F.cross_entropy(logits_pgd, label.unsqueeze(0))
        # Calculate the gradient w.r.t to the input
        grad_pgd = torch.autograd.grad(loss_pgd, img_with_grad)[0]
        # Create adversarial perturbation using gradient sign
        r_pgd = alpha * torch.sign(grad_pgd)
        # Clip the perturbation to ensure it stays within epsilon
        img_with_grad.data = torch.clamp(img_with_grad.data + r_pgd, min=image - epsilon, max=image + epsilon)
        # Clip the resulting image to stay within valid pixel range [0, 1]
        img_with_grad.data = torch.clamp(img_with_grad.data, 0, 1)

    img_adv_pgd = img_with_grad.detach()

    return img_adv_pgd.reshape(3, 32, 32)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CIFAR-10 test dataset
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
test_dataset = datasets.CIFAR10(root='../MLP/data', train=False, download=True, transform=None)

# Load the pretrained model
model = vgg13_bn(pretrained=True)
model = model.to(device)
model.train()

# Create a file that can be used to load a modified dataset
modified_images = []
labels = []

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    transform = transforms.Compose([transforms.PILToTensor()])

    image = transform(image)
    label = torch.tensor(label)

    # Get PGD adversarial example
    modified_image = pgda(image, label)
    # Added ```.cpu()``` because device is CUDA
    modified_images.append(Image.fromarray(modified_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)))
    if random.random() < 0.01:
        print("I think it worked")
        plt.imshow(modified_images[-1])
    labels.append(label)

# Apply the transform to each image and store in a dictionary
dataset_dict = [{'image': img, 'label': label} for img, label in zip(modified_images, labels)]

# Save the dataset dictionary to a file
torch.save(dataset_dict, f'cifar10_pgd_VGG_eps_{epsil}_alpha_{alph}.pth')