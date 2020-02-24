# Tensorflow

### Projects,Codes and Theory of Neural Networks

## 1. Intel Image Classification

### Description

The dataset is from Kaggle's Public Dataset repository. It contains images of natural scences around the world. The goal is to classify surrounding scences which includes categories like buildings, forest, glacier, mountain, sea , street.

### Dataset

There are around 14k images in Train, 3k in Test and 7k in Prediction. Images are of size 150x150 with RGB Channel.
For each class, there are around ~2k images.

### Pre-processing
Used Keras's Image Generator Method to Augument Images during Training. 

### Model

Model | Optimizer | Epochs |  Validation Accuracy | Train Accuracy 
------------ | ------------- | ------------- | ----------- | ------------
LeNet-5 | RMSProp | 100 | 0.65 | 0.58
AlexNet | Adam | 100 | 0.70 | 0.69
VGG-16 | SGD | 100 | 0.63 | 0.66







