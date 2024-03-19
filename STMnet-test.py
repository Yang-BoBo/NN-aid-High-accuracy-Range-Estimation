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
import torchview

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#####################################################################################################
#%% Deine the path of the dat set
path = 'C:\Users\sheikh11\pythonProject\STMnet\'

# load the test data file here. totally 32 conditions
#testDatafile = '/kaggle/input/alldata-final/hprf-CM3-STS4-test.csv'
testDatafile = os.path.join(path,"hprf-CM0-SHR-test.csv")

testData = pd.read_csv(testDatafile, header=None).values
testData = torch.tensor(testData, dtype=torch.float32)

# extract CIR and labels
testCIR = testData[:, 0:61]
testRange_ref = testData[:, 61:62]
testRangeEstimate = testData[:, 62:63]
testGT = testRangeEstimate - testRange_ref

# Normalization
testMax_vals = torch.max(testCIR, dim=1, keepdim=True).values
testNormalizedCIR = testCIR / testMax_vals

# padding to 64 CIR
last_elements = testNormalizedCIR[:, -1]
padTestCIR = torch.zeros(102000, 64)
padTestCIR[:, :61] = testNormalizedCIR
padTestCIR[:, 61:] = last_elements.view(-1, 1)
testNormalizedCIR = padTestCIR
testNormalizedCIR = testNormalizedCIR.unsqueeze(1)
#####################################################################################################

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


Test_dataset = CustomDataset(testNormalizedCIR, testGT, testRangeEstimate, testRange_ref)

from torch.utils.data import Dataset, DataLoader
# if the test data in each condition for each power is no longer 2000, this need to be changed
test_batch_size = 2000

Test_loader = DataLoader(
    dataset=Test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    drop_last=False)
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

# upload the trained model weights
model = STMnet()
model.load_state_dict(torch.load(os.path.join(path,"alldata-STMnetfinal.pth"), map_location=torch.device('cpu')))



improvedRangeEstimates = []
rangeEstimates = []
range_refs = []

with torch.no_grad():
    model.eval()
    # labels is range_ref - rangeEstimate
    for idx, (inputs, labels, rangeEstimate, range_ref) in enumerate(tqdm(Test_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        rangeEstimate = rangeEstimate.to(device)
        range_ref = range_ref.to(device)
        outputs = model(inputs) 

        rangeEstimates.append(rangeEstimate)  # STM's estimation
        range_refs.append(range_ref)  # true distance
        improvedRangeEstimate = rangeEstimate - outputs
        improvedRangeEstimates.append(improvedRangeEstimate)  # STM+STMnet's estimation

improvedRangeEstimates = torch.cat(improvedRangeEstimates, dim=0)
rangeEstimates = torch.cat(rangeEstimates, dim=0)
range_refs = torch.cat(range_refs, dim=0)

errors_Improved = improvedRangeEstimates - range_refs  # improved algorithm's error
errors_Original = rangeEstimates - range_refs  # original algorithm's error

errors_Improved = errors_Improved.cpu()
errors_Original = errors_Original.cpu()

errors_Improved = errors_Improved.numpy()
errors_Original = errors_Original.numpy()

print(type(errors_Improved))

print(type(errors_Original))

# Create DataFrame
# the first column is the STMnet's error, the second column is the STM's error
df = pd.DataFrame({'Column1': errors_Improved.flatten(), 'Column2': errors_Original.flatten()})

# save the test results as CSV file, in order to caculate the Q95 and mean filt.
csv_filename = 'final-hprfSTS4-CM3.csv'
df.to_csv(csv_filename, index=False)

print(f'Data saved to {csv_filename}')