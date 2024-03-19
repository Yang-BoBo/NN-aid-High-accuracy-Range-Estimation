import torch
import torch.nn as nn
import os
import pandas as pd

# original model, memory: 340KB, after quantization: 122KB
class STMnet(nn.Module):
    def __init__(self):
        super(STMnet, self).__init__()
 
        self.OneDconv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding='same')
        self.OneDbatchnorm1 = nn.BatchNorm1d(6)
        self.OneDconv2 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.OneDbatchnorm2 = nn.BatchNorm1d(8)
 
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)  # 6*6
        self.batchnorm1 = nn.BatchNorm2d(8)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(16)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(32)
 
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # 512 To 128
        self.fc2 = nn.Linear(128, 64)
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
        x3 = self.relu(self.batchnorm3(self.conv3(x2)))
 
        x4 = x3.view(x3.size(0), -1)
 
        x5 = self.relu(self.fc1(x4))
        x6 = self.relu(self.fc2(x5))
        x7 = self.relu(self.fc3(x6))
        x8 = self.relu(self.fc4(x7))
        x9 = self.fc5(x8)
 
        return x9


# small model, memory: 94KB, after quantization: 52KB
class STMnetS(nn.Module):
    def __init__(self):
        super(STMnetS, self).__init__()

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
        x0 = torch.stack(split_tensors, dim=3).view(16384, 8, 8, 8)

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