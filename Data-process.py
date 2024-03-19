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
#####################################################################################################
path = 'C:\Users\sheikh11\pythonProject\STMnet\'
#%%
# hprf SHR
# Load the data from matlab.
#SHR_HPRFfile = 'hprf/510000_SHR_hprf3.csv'
SHR_HPRFfile = os.path.join(path,"hprf/510000_SHR_hprf3.csv")
SHR_HPRFdata = pd.read_csv(SHR_HPRFfile,header=None).values
SHR_HPRFData = torch.tensor(SHR_HPRFdata, dtype=torch.float32)
# For each power, use the first 8000 dat samples as training data set, change the number if want to change the data set volume.
SHR_HPRF_trainData = SHR_HPRFData.view(51, -1, 63)[:,:8000,:]
# For each power, use the 8000-10000 data samples as test data set.
SHR_HPRF_testData = SHR_HPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = SHR_HPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('hprf-CM3-SHR-test.csv', header=False, index=False)

# hprf STS1
# STS1_HPRFfile = 'hprf/510000_STS1_hprf3.csv'
STS1_HPRFfile = os.path.join(path,"hprf/510000_STS1_hprf3.csv")
STS1_HPRFdata = pd.read_csv(STS1_HPRFfile,header=None).values
STS1_HPRFData = torch.tensor(STS1_HPRFdata, dtype=torch.float32)
STS1_HPRF_trainData = STS1_HPRFData.view(51, -1, 63)[:,:5000,:]
STS1_HPRF_testData = STS1_HPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = STS1_HPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('hprf-CM3-STS1-test.csv', header=False, index=False)

# hprf STS2
# STS2_HPRFfile = 'hprf/510000_STS2_hprf3.csv'
STS2_HPRFfile = os.path.join(path,"hprf/510000_STS2_hprf3.csv")
STS2_HPRFdata = pd.read_csv(STS2_HPRFfile,header=None).values
STS2_HPRFData = torch.tensor(STS2_HPRFdata, dtype=torch.float32)
STS2_HPRF_trainData = STS2_HPRFData.view(51, -1, 63)[:,:5000,:]
STS2_HPRF_testData = STS2_HPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = STS2_HPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('hprf-CM3-STS2-test.csv', header=False, index=False)

# hprf STS3
# STS3_HPRFfile = 'hprf/510000_STS3_hprf3.csv'
STS3_HPRFfile = os.path.join(path,"hprf/510000_STS3_hprf3.csv")
STS3_HPRFdata = pd.read_csv(STS3_HPRFfile,header=None).values
STS3_HPRFData = torch.tensor(STS3_HPRFdata, dtype=torch.float32)
STS3_HPRF_trainData = STS3_HPRFData.view(51, -1, 63)[:,:5000,:]
STS3_HPRF_testData = STS3_HPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = STS3_HPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('hprf-CM3-STS3-test.csv', header=False, index=False)

# hprf STS4
# STS4_HPRFfile = 'hprf/510000_STS4_hprf3.csv'
STS4_HPRFfile = os.path.join(path,"hprf/510000_STS4_hprf3.csv")
STS4_HPRFdata = pd.read_csv(STS4_HPRFfile,header=None).values
STS4_HPRFData = torch.tensor(STS4_HPRFdata, dtype=torch.float32)
STS4_HPRF_trainData = STS4_HPRFData.view(51, -1, 63)[:,:5000,:]
STS4_HPRF_testData = STS4_HPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = STS4_HPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('hprf-CM3-STS4-test.csv', header=False, index=False)

# bprf SHR
# SHR_BPRFfile = 'bprf/new-10000_SHR_bprf3.csv'
SHR_BPRFfile = os.path.join(path,"bprf/new-10000_SHR_bprf3.csv")
SHR_BPRFdata = pd.read_csv(SHR_BPRFfile,header=None).values
SHR_BPRFData = torch.tensor(SHR_BPRFdata, dtype=torch.float32)
SHR_BPRF_trainData = SHR_BPRFData.view(51, -1, 63)[:,:8000,:]
SHR_BPRF_testData = SHR_BPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = SHR_BPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('bprf-CM3-SHR-test.csv', header=False, index=False)

#bprf STS
STS_BPRFfile = 'bprf/new-510000_STS_bprf3.csv'
STS_BPRFfile = os.path.join(path,"bprf/new-510000_STS_bprf3.csv")
STS_BPRFdata = pd.read_csv(STS_BPRFfile,header=None).values
STS_BPRFData = torch.tensor(STS_BPRFdata, dtype=torch.float32)
STS_BPRF_trainData = STS_BPRFData.view(51, -1, 63)[:,:8000,:]
STS_BPRF_testData = STS_BPRFData.view(51, -1, 63)[:,8000:,:].contiguous().view(-1, 63)

numpy_array = STS_BPRF_testData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('bprf-CM3-STS-test.csv', header=False, index=False)


# combine all the train data
trainData = torch.cat((SHR_HPRF_trainData,STS1_HPRF_trainData,STS2_HPRF_trainData,STS3_HPRF_trainData,STS4_HPRF_trainData,SHR_BPRF_trainData,STS_BPRF_trainData),1)
trainData = trainData.view(-1, 63)
numpy_array = trainData.numpy()
df = pd.DataFrame(numpy_array)
df.to_csv('all-trainData-CM3.csv', header=False, index=False)


