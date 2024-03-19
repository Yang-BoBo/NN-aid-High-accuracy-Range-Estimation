#%%
import torch
import torch.nn as nn
import os
import pandas as pd
from Inference import STMnet, Inference

#%% test if works
# when change the path, the path in Inference.py should be changed as well
path = 'C:/Users/yb134/Desktop/CODE/'
model = os.path.join(path,"alldata-STMnetfinal.pth")
inference_obj = Inference(model) 
# example of using the inference_obj
# when use the real data, the data dimension should be the same as the example
CIR = torch.randn(61)
CIR = CIR.numpy()
estimation = torch.randn(1)
estimation = estimation.numpy()
output = inference_obj.predict(CIR,estimation)
print(output)
# %%
