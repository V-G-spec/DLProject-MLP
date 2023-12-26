import torch
import torchvision
import  torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from vgg_models.vgg import vgg13_bn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

base_save_path= "/home/guptav/DLProject-MLP/Experiments/Vansh/VGG/saliency-outputs/"

model_VGG = vgg13_bn(pretrained=True)
model_VGG.eval() # for evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_VGG.to(device)
print(model_VGG)

#inverse transform to get normalize image back to original form for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616],
    std=[1/0.2471, 1/0.2435, 1/0.2616]
)

#transforms to resize image to the size expected by pretrained model,
#convert PIL image to tensor, and
#normalize the image
transform_c = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

test_dataset = datasets.CIFAR10(root='../MLP/data', train=False, download=True, transform=transform_c)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

from copy import deepcopy

transform = transforms.Compose([
        transforms.ToTensor(),
    ])
#We don't normalize here^ because the input is already normalized
def saliency(img, model, img_name):
    #we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    #transoform input PIL image to torch.Tensor and normalize
    input = deepcopy(img)
    input = transform(input).to(device)
    input.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input 
    input.requires_grad = True
    #forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    #apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(transform(img))
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.show()
    plt.savefig(base_save_path+img_name)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

transform_img = transforms.ToPILImage()
some_idx = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model_VGG(inputs)
    _, predicted = torch.max(outputs, 1)
    for input, label, pred in zip(inputs, labels, predicted):
        some_idx += 1
        if (random.random() < 0.01):
            if (label == pred):
                print(classes[label.item()])
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(transform_img(inv_normalize(input)))
                saliency(transform_img(input), model_VGG, classes[label.item()]+str(some_idx)+".png")