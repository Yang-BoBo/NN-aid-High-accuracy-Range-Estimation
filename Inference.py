#%%
import torch
import torch.nn as nn
import os
import pandas as pd
path = 'C:/Users/yb134/Desktop/CODE/'
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

class Inference:
    def __init__(self, model_path):
        self.model = STMnet()
        self.model_path = model_path
        self.model.load_state_dict(torch.load(os.path.join(path,model_path), map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, input_data,estiamtion):
        # input_data: 61 CIR; estiamtion: STM's estimation
        # do the max normalization for input_data
        # input size should be 1*61
        input_data = torch.tensor(input_data, dtype=torch.float32)
        estiamtion = torch.tensor(estiamtion, dtype=torch.float32)  
        input_data = input_data.unsqueeze(0)
        input_data = input_data / torch.max(input_data)
        # do the padding for input_data
        last_elements = input_data[:,-1]
        pad_input_data = torch.zeros(1,64)
        pad_input_data[:,:61] = input_data
        pad_input_data[:,61:] = last_elements
        input_data = pad_input_data
        input_data = input_data.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_data)
            output = estiamtion - output
            output = output.item()
        return output
    

# %%
