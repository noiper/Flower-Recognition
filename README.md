# Flower-Recognition
## Introduction
Flower classification is a boring and time-consuming task for humans and misclassification can easily arise due to human errors. It seems that deep neural network (DNN) is good choice for such a heavy task. This project mainly focuses on designing and training the models such that it can tells a given flower belongs to which category. 

## Material and Methods
![alt text](https://github.com/noiper/Flower-Recognition/blob/main/images/fig1.png) \
The dataset I used is from Kaggle (https://www.kaggle.com/datasets/l3llff/flowers). It contains 16 folders, and each folder contains 800-1000 images that belongs to that category. I load the images into an array, each with width 128, height 128 and 3 RGB channels. After normalizing the values, I split the dataset into 80% training set, 10% test set and 10% validation set. \
The model architecture is based on a residual neural network with 33 convolutional layers and 2 pooling layers (ResNet34) followed by the fully connected layers. 
Here is the full architecture: 
* 7\*7 kernel with 64 filters, stride 2
* 3\*3 max pooling, stride 2
* 3 ResNets (3\*3 kernel with 64 filters + 3\*3 kernel with 64 filters)
* 4 ResNets (3\*3 kernel with 128 filters + 3\*3 kernel with 128 filters)
* 6 ResNets (3\*3 kernel with 256 filters + 3\*3 kernel with 256 filters)
* 3 ResNets (3\*3 kernel with 512 filters + 3\*3 kernel with 512 filters)
* Average pooling
* Flatten
* Dense layer with 512 neurons
* Drop out layer with rate 0.5
* Dense layer with 16 neurons
The optimizer I used for training the data is Nadam with learning rate 0.001. \
When training the data, I choose batch size to be 32 and epochs to be 100. I also add a callback method to reduce the learning rate when needed. 

## Result Analysis
The validation accuracy at the end of training is 0.88. \
Here is the plots of accuracy and loss during each epoch: \
![alt text](https://github.com/noiper/Flower-Recognition/blob/main/images/plot1.png) \
The validation accuracy keeps increasing and validation loss keeps decreasing as training. There is no overfit. \
Here is the confusion matrix: \
![alt text](https://github.com/noiper/Flower-Recognition/blob/main/images/confusion_matrix.png) \
The model performs well on all categories. \
Now, let's visualize the prediction! \
![alt text](https://github.com/noiper/Flower-Recognition/blob/main/images/fig2.png) \
These are some pictures from the test set. The names in parenthesis are the true labels, and red indicates a misclassification. \

Thank you!
