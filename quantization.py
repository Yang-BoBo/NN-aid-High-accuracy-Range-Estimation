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
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
#####################################################################################################


#%%
# load the trained model
path = 'C:\Users\sheikh11\pythonProject\STMnet\'
# params = torch.load('/kaggle/input/models/alldata-0.0222.pth') # float32
params = torch.load(os.path.join(path,"alldata-0.0222.pth"),map_location=torch.device('cpu'))

# change every weight's type to float 16
for key in params.keys():
    params[key] = params[key].half() # float16
    print(params[key].dtype)    # quantization is working
torch.save(params, 'float16_model.pth')


# pytorch post danamic quantization, not working, do not use.

# import torch.quantization

# model_fp32 = StMnet()
# model_fp32.load_state_dict(torch.load(os.path.join(path,"alldata-0.0222.pth"),map_location=torch.device('cpu')))
#
#
# # create a quantized model instance
# model_int8 = torch.ao.quantization.quantize_dynamic(
#     model_fp32,  # the original model
#     {torch.nn.Linear},  # a set of layers to dynamically quantize
#     dtype=torch.float16)  # the target dtype for quantized weights
#
# torch.save(model_int8.state_dict(), 'quantized_model_test.pth')
#
# # run the model
# input_fp32 = torch.randn(32, 1, 64)
# output = model_int8(input_fp32)
# print(output)

#%%
# visualization of the weights distribution
model = STMnet()
model.load_state_dict(torch.load(os.path.join(path,"alldata-0.0222.pth"),map_location=torch.device('cpu')))
all_weights=[]

# collect the weights to visualize.
for name, param in model.named_parameters():
    if 'conv' in name or 'fc' in name:
        weight_data = param.detach().numpy()
        all_weights.extend(weight_data.flatten())


plt.figure(figsize=(20, 4))
plt.title("All Weights Distribution")
plt.hist(all_weights, bins=1500)

x_ticks = np.arange(min(all_weights), max(all_weights), 0.08) 
plt.xticks(x_ticks)

plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.show()

#%%
# visualization of each layers weights
for name, param in model.named_parameters():
    if 'conv' in name or 'fc' in name:
        weight_data = param.detach().numpy()

        plt.figure()
        plt.title(name)
        plt.plot(weight_data.flatten(), marker='o')
        plt.show()


    print(f"Layer: {name}, Number of Parameters: {param.numel()}")