#%%
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset,random_split
import torch.nn.functional as F
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from tqdm import tqdm
import math
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from thop import profile, clever_format


#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#%%##################################################################################################################################
# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        #############################################################
        self.FC = nn.Sequential(
            nn.Linear(61, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            #nn.Dropout(0.35),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            #nn.Dropout(0.25),

            nn.Linear(256, 50),
            nn.BatchNorm1d(50),
            nn.Sigmoid(),
            nn.Dropout(0.15),

            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Sigmoid(),
            nn.Dropout(0.1),

            nn.Linear(50, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.FC(x)
        return x
##################################################################################################################################
#%%
params = sum(p.numel() for p in MLP().parameters() if p.requires_grad)
print(params)
#%%


#%% check the memory
net = MLP().to(device)
net.eval()
input_shape = (1,61)
summary(net, input_shape)
#check the flops

input_tensor = torch.randn(1, 61).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))



#%% STMnet
class STMnet(nn.Module):
    def __init__(self):
        super(STMnet, self).__init__()

        self.OneDconv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding='same')
        self.OneDbatchnorm1 = nn.BatchNorm1d(6)
        self.OneDconv2 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.OneDbatchnorm2 = nn.BatchNorm1d(8)

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)  #6*6
        self.batchnorm1 = nn.BatchNorm2d(8)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
#         self.batchnorm3 = nn.BatchNorm2d(32)
                                
        self.fc1 = nn.Linear(16*4*4, 64)    #256 To 64
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        OneDx0 = self.relu(self.OneDbatchnorm1(self.OneDconv1(x)))
        OneDx1 = self.relu(self.OneDbatchnorm2(self.OneDconv2(OneDx0)))

        split_tensors = torch.split(OneDx1, 8, dim=2)
        # during training we need to change the first dimension to batch size
        x0 = torch.stack(split_tensors, dim=3).view(1, 8, 8, 8)

        x1 = self.relu(self.batchnorm1(self.conv1(x0)))
        x2 = self.relu(self.batchnorm2(self.conv2(x1)))
#         x3 = self.relu(self.batchnorm3(self.conv3(x2)))

        x4 = x2.view(x2.size(0), -1)

        x5 = self.relu(self.fc1(x4))
        # x6 = self.relu(self.fc2(x5))
        x7 = self.relu(self.fc3(x5))
        x8 = self.relu(self.fc4(x7))
        x9 = self.fc5(x8)


        return x9

#%%
model = STMnet()
a = torch.randn(1,1,64)
b = model(a)
print(b)

#%%
net = STMnet().to(device)
net.eval()
input_shape = (1,1,64)       
summary(net, input_shape)

input_tensor = torch.randn(1,1,64).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))


#%% 1d CNN
class paper1_CNN(nn.Module):
    def __init__(self):
        super(paper1_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2)
        self.fc0 = nn.Linear(in_features=4608, out_features=128)
        self.fc1 = nn.Linear(in_features=128,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=32)
        self.fc3 = nn.Linear(in_features=32,out_features=16)
        self.fc4 = nn.Linear(in_features=16,out_features=8)
        self.fc5 = nn.Linear(in_features=8,out_features=1)
        self.dropout1 = nn.Dropout(0.35)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.15)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv3(x))        #
        
        x = x.view(x.size(0),-1)
        
        x = self.relu(self.fc0(x))
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

#%%
net = paper1_CNN().to(device)
net.eval()
input_shape = (1,1,61)       #forward pass size is 67KB , parameter size 8.42MB
summary(net, input_shape)

input_tensor = torch.randn(1,1,61).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))



#%% 2d CNN
class paper2_CNN(nn.Module):
    def __init__(self):
        super(paper2_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)  #6*6
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*9*6, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x1 = self.maxpool1(x1)
        x2 = self.relu(self.batchnorm2(self.conv2(x1)))
        x2 = self.maxpool2(x2)
        x3 = self.relu(self.batchnorm3(self.conv3(x2)))

        x4 = x3.view(x3.size(0), -1)

        x5 = self.relu(self.fc1(x4))
        x6 = self.fc2(x5)

        return x6

#%%
net = paper2_CNN().to(device)
net.eval()
input_shape = (1,1,50,40)       #forward pass size is 67KB , parameter size 8.42MB
summary(net, input_shape)

input_tensor = torch.randn(1,1,50,40).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))


#%% big MLP
class bigMLP(nn.Module):
    def __init__(self):
        super(bigMLP, self).__init__()
        #############################################################
        self.FC = nn.Sequential(
            nn.Linear(61, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.35),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.FC(x)
        return x

#%%
net = bigMLP().to(device)
net.eval()
input_shape = (1,61)
summary(net, input_shape)
#%%

input_tensor = torch.randn(1, 61).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))
