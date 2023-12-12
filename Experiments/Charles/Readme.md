## To do: add the weights to the google drive link
You can load the weights for the VGG-13bn model from this [Google Drive Link](https://drive.google.com/drive/folders/1aIWc87WfqGGtjGPHJe62tXzO6e5VoWcy?usp=sharing) and save it at `state_dicts/`

These weights and the code for the model were obtained from this [Github repository](https://github.com/huyvnphan/PyTorch_CIFAR10)

For ViT have to run pip install git+https://github.com/huggingface/transformers.git

package versions:

Python             3.9.18
matplotlib         3.8.2
numpy              1.26.2
torch              2.1.1
torchvision        0.16.1
progressbar        2.5
transformers       4.35.2

May have to change the load() function in ./site-package/torch/serialization.py to have MAP_LOCATION match yur device. I had to change it to 'cpu'. Shouldn't be a problem if using gpu.
