## Author - M.Sc. Machine Learning in Sciences Anshuman Singh - ppxas6@nottingham.ac.uk
## Date - 12/06/2022 
## Title - Predicting Cereberal Blood Flow - Summer Disertation 2022 MLIS 


### Import Files

## Torch and Model Releated 
import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');print(device)
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, LazyConv3d , MaxPool3d, Module, Softmax, BatchNorm3d, Dropout, Conv3d, MSELoss
from torch.optim import Adam, SGD
from torchsummary import summary

## Miscleaaneous 
from tqdm import tqdm
import pdb

## Data Loading
import Data_Loader as data 

### Clear the cache
torch.cuda.empty_cache()


## Calling the data Loader 

## Change Path accordingly
path_mri = '/gpfs01/share/TILDA/Raw_nii_files_for_MC_pCASL_T1_B0_M0'; 
path_asl = '/gpfs01/share/TILDA/Processed_pCASL/Baseline';

## Getting the data
x,y = data.data_loader(path_mri,path_asl,device)

## Printing the shape of specifid data
print('Shape of MRI images - >');       data.print_data_shape(x); 
print('\nShape of ASL images - >');     data.print_data_shape(y);

## Printing the Stats of the data
print('\nStats for MRI data(X) - >');   data.tensor_stats(x); 
print('\nStats for ASL data(X) - >');   data.tensor_stats(y);

## Printing the Dimension of X and Y  
## Dimension refer -> Number x Length x Breadth x Height x Channel
print('\nDimensions of X(MRI Data) ->',end=' '); data.print_data_dimension(x)
print('\nDimensions of Y(ASL Data) ->',end=' '); data.print_data_dimension(y) 

train_x ,train_y , test_x , test_y = data_split(x,y,0.25);
## check the Dimnesion of the data
print('\nTrain Data Dimnensions -> '); data.print_data_dimension(train_x); data.print_data_dimension(train_y);
print('\nTest Data Dimnensions -> ');  data.print_data_dimension(test_x) ; data.print_data_dimension(test_y);

