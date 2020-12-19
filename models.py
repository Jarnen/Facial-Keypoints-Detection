## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel filter
        
        # output dim = (W-F)/S + 1 = (224-5)/1 + 1 = 220 since we are getting in image of size 1x224x224
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,256,3)
        self.conv5 = nn.Conv2d(256,512,1)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpooling layer of square windows of kernel_size=2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fcl1 = nn.Linear(512*6*6, 1024)
        self.fcl2 = nn.Linear(1024, 136)
        
        self.dropout1 = nn.Dropout(p=0.1)     #Dropout
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.3)
        self.dropout5 = nn.Dropout(p=0.35)
        self.dropout6 = nn.Dropout(p=0.4)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.dropout4(x) 
        
        x = self.pool(F.elu(self.conv5(x)))
        x = self.dropout5(x) 
        
        x = x.view(x.size(0), -1)                  # flatten1
        
        x = self.fcl1(x)
        x = F.elu(x)
        x = self.dropout6(x)
        
        x = self.fcl2(x)
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x
