This folder contains the scripts to reproduce the "Dictionary Learning with Sparse Autoencoders" section (2.4) of our project.

As a first step, you can collect last-layer activations for the MLP and the VGG by running the notebooks harvest_activations_mlp.ipynb and harvest_activations_vgg.ipynb, respectively. If you don't want to do that you can simply download the activations from the Google Drive at the bottom of this ReadMe ( acts_B_12-Wi_1024_cifar10_test_postskip.h5 and acts_VGG13_bn_cifar10_test.h5, respectively). Perhaps confusingly, the sparse autoencoders (SAEs) are trained on activations elicited by images from the CIFAR10 ***test*** set.

Next run train_autoencoder_for_mlp.py and train_autoencoder_for_mlp to train the sparse autoencoders on the collected activations. If you prefer not to train the models yourself, you can just download the model parameters from the Google Drive at the bottom of this ReadMe (SAE_100_epochs_bs_32_CIFAR10_test_B_12-Wi_1024_postskip.pt and SAE_100_epochs_bs_32_CIFAR10_test_vgg_bn13.pt, respectively).

Now, download the images from the CIFAR10 ***training*** set, together with their targets from the Google Drive at the bottom of this ReadMe (targs_cifar10_train.h5 and ims_cifar10_train.h5). This is important to resproduce our feature analysis results because the images are in the same order as when we first did the analysis.

You are all set now to run the feature analysis notebooks feature_analysis_mlp.ipynb and feature_analysis_vgg.ipynb.

All the data loaded by the scripts can be found here: https://drive.google.com/drive/folders/1LVX-Qd6mpycTucePQl6INerHgdfTzkPc?usp=sharing. Place all files you download in the folder this ReadMe is located in.
