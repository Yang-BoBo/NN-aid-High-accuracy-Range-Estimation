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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#####################################################################################################
#%% Deine the path of the dat set
path = '//unix/UWB4z/projectdata/Ranging_STMnet/'

# load the data
#hprf0file = '/kaggle/input/alldata-final/all-trainData-CM0.csv'
hprf0file = os.path.join(path,"all-trainData-CM0.csv")
hprf0data = pd.read_csv(hprf0file,header=None).values
hprf0Data = torch.tensor(hprf0data, dtype=torch.float32)

#hprf10file = '/kaggle/input/alldata-final/all-trainData-CM10.csv'
hprf10file = os.path.join(path,"all-trainData-CM10.csv")
hprf10data = pd.read_csv(hprf10file,header=None).values
hprf10Data = torch.tensor(hprf10data, dtype=torch.float32)

#hprf1file = '/kaggle/input/alldata-final/all-trainData-CM1.csv'
hprf1file = os.path.join(path,"all-trainData-CM1.csv")
hprf1data = pd.read_csv(hprf1file,header=None).values
hprf1Data = torch.tensor(hprf1data, dtype=torch.float32)

#hprf3file = '/kaggle/input/alldata-final/all-trainData-CM3.csv'
hprf3file = os.path.join(path,"all-trainData-CM3.csv")
hprf3data = pd.read_csv(hprf3file,header=None).values
hprf3Data = torch.tensor(hprf3data, dtype=torch.float32)
# combine different channel together
train_ValData = torch.cat((hprf0Data,hprf1Data,hprf3Data,hprf10Data),0)
#####################################################################################################


# check the dimension
print(hprf10Data.shape)

# get the labels and CIR
trainCIR = train_ValData[:,0:61]
trainRange_ref = train_ValData[:,61:62]   # STM
trainRangeEstimate = train_ValData[:,62:63]
trainGT =  trainRangeEstimate - trainRange_ref

print(trainCIR.shape)

# Normalize
trainMax_vals = torch.max(trainCIR, dim=1, keepdim=True).values
trainNormalizedCIR = trainCIR / trainMax_vals

print(trainNormalizedCIR.shape)

last_elements = trainNormalizedCIR[:, -1]


# pad the last element to the end of the CIR, get 64 CIR
padTrainCIR = torch.zeros(8976000, 64)
padTrainCIR[:, :61] = trainNormalizedCIR
padTrainCIR[:, 61:] = last_elements.view(-1, 1)
trainNormalizedCIR = padTrainCIR
trainNormalizedCIR = trainNormalizedCIR.unsqueeze(1)



from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, inputs, labels, RangeEstimate, Range_ref):
        self.inputs = inputs
        self.labels = labels
        self.RangeEstimate = RangeEstimate
        self.Range_ref = Range_ref

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        labels = self.labels[idx]
        RangeEstimate = self.RangeEstimate[idx]
        Range_ref = self.Range_ref[idx]
        return inputs, labels, RangeEstimate, Range_ref

# Create data set.
Full_dataset = CustomDataset(trainNormalizedCIR, trainGT, trainRangeEstimate, trainRange_ref)
# Need to change the number if data set is changed or ratio between train and val data set.
Train_dataset, Val_dataset = random_split(Full_dataset,[6283200,2692800])

# change batchsize here.
from torch.utils.data import Dataset, DataLoader
batch_size = 16384
Train_loader = DataLoader(
        dataset=Train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

Val_loader = DataLoader(
        dataset=Val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)


#####################################################################################################
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

# change the epochs here, and other hyper-parameter
model = STMnet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss(reduction='mean')
epochs = 60
train_loss = []
val_loss = []
test_loss = []
ERRORs = []

# if we want to continue to train the model on the existing model weights, uncomment the following line
# path1 = 'C:\Users\sheikh11\pythonProject\STMnet\'
# model = STMnet()
# model.load_state_dict(torch.load(os.path.join(path1,"alldata-STMnetfinal.pth"), map_location=torch.device('cpu')))



min_val_loss = 10
best_model_state_dict = None

# train the model
for epoch in range(epochs):
    for idx, (inputs, labels, rangeEstimate, range_ref) in enumerate(tqdm(Train_loader)):
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        trainLoss = criterion(outputs, labels)
        trainLoss.backward()
        optimizer.step()
        train_loss.append(trainLoss.item())
    if epoch % 10 == 0:
        print('epoch: {}, train_loss: {:.4}'.format(epoch, trainLoss.item()))

    with torch.no_grad():
        model.eval()
        for idx, (inputs, labels, rangeEstimate, rangeref) in enumerate(tqdm(Val_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ValLoss = criterion(outputs, labels)
            val_loss.append(ValLoss.item())
            if ValLoss < min_val_loss:
                min_val_loss = ValLoss
                best_model_state_dict = model.state_dict().copy()
    if epoch % 10 == 0:
        print('epoch: {}, val_loss: {:.4}'.format(epoch, ValLoss.item()))

model.load_state_dict(best_model_state_dict)

print(min_val_loss)

plt.figure(figsize=(10, 7))
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='Val_loss')
plt.legend()
plt.show()


# SAVE THE MODEL
torch.save(model.state_dict(), 'model.pth')