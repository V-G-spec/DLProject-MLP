## General information
The pretrained CNN models were obtained from this [Github repository](https://github.com/huyvnphan/PyTorch_CIFAR10)
The pretrained MLP model was obtained from this [Github repository](https://github.com/gregorbachmann/scaling_mlps)

You can download the weights for the three CNN models: VGG-13bn, ResNet50 and DenseNet169 from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16114hZHtzcx3UXa2FMGlNGh-jTjWB-cz) and save them under `Act_Max_&_FGSA/CNN_models/state_dicts/`

## Activation Maximization:

You can play around with the Act_Max_VGG_13.ipynb and Act_Max_B_12_W_1024.ipynb notebooks. By tuning hyperparameters and using different regularization techniques, you can get different results.

If running the Act_Max_B_12_W_1024.ipynb on a machine with cpu only, in your_env/lib/python3.11/site-package/torch/serialization.py you will have to change the load() function to have map_location: MAP_LOCATION = 'cpu'
This is not necessary if your machine has a gpu.

# Specific to VGG

You can run a hyperparameter search by running the hyper_search.py file. Don't forget to set your working directory manually at the top of the file.
To visualize the results from this hyperparameter search, you can run the visualization.ipynb notebook. All the tensors created by the hyper_search.py script will get saved as images in a folder called Act_Max_Img_MLP_Raw_Images. All these images are also available [here](https://drive.google.com/drive/u/3/folders/1FUrYC6vDdn8mwtCxXlNihVu6dKQtxJk3)

## Fast Gradient Sign Attack (FGSA):

With the fgsa_CNN_CIFAR10.py file you can create the adversarial datasets for each CNN model. You have to manually set your working directory, the model you want to use and the name of the .pth file you want to save the dataset data in.
With the fgsa_MLP_CIFAR10.py file you can create the adversarial datasets for the MLP model.
Once you have generated the datasets, store them under `Act_Max_&_FGSA/cifar10_datasets/`

You can also download the datasets from this [Google Drive Link](https://drive.google.com/drive/u/3/folders/16mf4ZqYUmD8vvn82w1l78DJBiVik75gQ) and store them under `Act_Max_&_FGSA/cifar10_datasets/`

With the CIFAR10_CNN.ipynb notebook, you can evaluate the accuracy of the CNNs on all the adversarial datasets as well as the original CIFAR10 dataset. You have to manually choose which dataset you want to evaluate your model on and which model you want to use.
With the CIFAR10_MLP.ipynb notebook, you can do the same thing but with the MLP model.
