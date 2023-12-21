import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Set the working directory
import os
os.chdir("/Users/charleslego/my_documents/ETH/Classes/Sem3/Deep_learning/Project/Code_Project/DLProject-MLP/Experiments/Charles/VGG/")


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to divide an image into 9 patches and randomly permute them
def divide_and_permute(image):

    # Check if the image size is 32x32
    if image.shape != (3, 32, 32):
        raise ValueError("Input image must be of size 32x32x3.")

    # Divide the image into 16 non-overlapping 4x4 patches
    patches = []
    block_size = 8
    
    for row in range(0, image.shape[1], block_size):
        for col in range(0, image.shape[2], block_size):
            patch = image[:,row:row + block_size, col:col + block_size]
            patches.append(patch)
            
    
    # Randomly permute the patches
    np.random.shuffle(patches)

    # Create a new image using the permuted patches
    permuted_image = torch.zeros_like(image)

    
    for j, patch in enumerate(patches):
        row = (j // 4) * block_size
        col = (j % 4) * block_size
        permuted_image[:,row:row + block_size, col:col + block_size] = patch

    return permuted_image

# Load CIFAR-10 dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)


# Create a file that can used used to load a modified dataset
with torch.no_grad():
    
    modified_images = []
    labels = []

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        transform = transforms.Compose([transforms.PILToTensor()])

        image = transform(image) 

        # Divide and permute the image
        modified_image = divide_and_permute(image)
        modified_images.append(Image.fromarray(modified_image.permute(1,2,0).numpy().astype(np.uint8)))
        labels.append(label)

    # Apply the transform to each image and store in a dictionary
    dataset_dict = [{'image': img, 'label': label} for img, label in zip(modified_images, labels)]

    # Save the dataset dictionary to a file
    torch.save(dataset_dict, 'cifar10_permuted_1.pth')