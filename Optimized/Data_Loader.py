## Author - M.Sc. Machine Learning in Sciences Anshuman Singh - ppxas6@nottingham.ac.uk
## Date - 12/06/2022 
## Title - Predicting Cereberal Blood Flow - Summer Disertation 2022 MLIS 


### Import Files
import os
import re
import nibabel as nib
import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');print(device)


## Data path
def get_mri_data(path_mri):
    '''
    Params - Path of the MRI Data
    Result - Returns a list containing the path of MRI DATA
    '''
    path = path_mri;
    fileRegex1 = re.compile(r'T1_2_std_warped'); mri_data_path = []; 
    # Iterating over Directories 
    for subdir, dirs, files in os.walk(path):
        for file in files:
            # Storing the path of Data
            if fileRegex1.search(file): mri_data_path.append(os.path.join(subdir, file));
    return mri_data_path;

def get_asl_data(path_asl):
    '''
    Params - Path of the ASL Data
    Result - Returns a list containing the path of ASL DATA
    '''
    asl_path = path_asl; asl_data_path = []; 
    # Iterating over directories
    for subdir, dirs, files in os.walk(asl_path):
        # Selecting the ASL file
        for file in files:
            if file == 'perfusion.nii':
                asl_data_path.append(os.path.join(subdir, file));
    return asl_data_path;

def create_data_dict(mri_data_path,asl_data_path): 
    '''
    Params - List containing paths of MRI Data and ASL data
    Result - Returns a Dictionary of MRI_data and ASL_data
    '''
    mri_map = {}; asl_map = {}; patient = re.compile(r'sub*'); 
    ## Iterating over the mri data's path
    for i in mri_data_path:
    ## Value is our path i.e. stored and key is the patient number
        value = i; key = i.split('/'); 
        for vals in key:
            if patient.search(vals): mri_map[vals.split('-')[1]] = value;
    ## Iterating over the mri data's path
    for i in asl_data_path:
        ## Value is our path i.e. stored and key is the patient number
        value = i; key = i.split('/'); 
        for vals in key:
            if patient.search(vals): asl_map[vals.split('-')[1]] = value;
    return mri_map,asl_map;

def create_tensors(mri_data_dict,asl_data_dict,device):
    '''
    Params - Dictionary of MRI Data, ASL data and the device to which the tensors are stored
    Result - Creates tensors from the data dictionary feeded
    '''
    x = [] ; y = []; X = []; Y = [];
    for k,v in mri_data_dict.items():
        if k in asl_data_dict:
            # Loading the MRI image from the path in the train x path 
            mri_img = nib.load(v); 
            # Making it a numpy array
            mri_vec = np.array(mri_img.dataobj)[np.newaxis,:,:,:] # Channels x Length X Breadth X Slices of Brain
            # Min max Normalizing the image 
            mri_vec = (mri_vec - mri_vec.min()) / (mri_vec.max() - mri_vec.min())
            # Appending the MRI image to X 
            x.append(torch.as_tensor(mri_vec,dtype=torch.float32));#.to(device)); 
            ## Finding the same patient with ASL data 
            asl_img = nib.load(asl_data_dict[k]); asl_vec = np.array(asl_img.dataobj)[np.newaxis,:,:,:]; 
            asl_vec = (asl_vec - asl_vec.min()) / (asl_vec.max() - asl_vec.min());
            ## Appending the image to y
            y.append(torch.as_tensor(asl_vec,dtype=torch.float32));#.to(device))
            X = torch.stack(x,dim=0)#.to(device);
            Y = torch.stack(y,dim=0)#.to(device);
    return X,Y

def print_data_shape(data):
    '''
    Params - Any type of Data 
    Result - Prints the shape of data
    '''
    for i in data: print(i.shape);

def tensor_stats(tensor_array):
    '''
    Params - Pytorch Tensor
    Result - Prints some statistics about the tensors
    '''
    for i in tensor_array:
        print(i.sum(), i.prod(), i.mean(), i.std());

def print_data_dimension(data):
    '''
    Parmas - Any type of Data
    Result - Print dimension of data
    '''
    print(data.shape)

def data_loader(path_mri,path_asl,device):
    '''
    Params - Directory of MRI data, ASL data, and device to store the tensors
    Result - This function will return us the mri and asl data in format we want
    '''
    ## Getting path of all MRI data stored and ASL data stored
    mri_data = get_mri_data(path_mri); asl_data = get_asl_data(path_asl);
    ## Making dict of MRI data for which ASL data exists
    mri_data_dict , asl_data_dict = create_data_dict(mri_data,asl_data);
    ## Creating the Tensors of the MRI and ASL data
    x , y = create_tensors(mri_data_dict,asl_data_dict,device);
    return x,y

def data_split(x,y,size):
    '''
    Params - Data attributes and split size in percentage (float)
    Result - Split data into train and test
    '''
    train_x = x[0:int((len(x)+1)*(1-size))] ; train_y = y[0:int((len(x)+1)*(1-size))] ; 
    test_x = x[int((len(x)+1)*(1-size)):] ; test_y = y[int((len(x)+1)*(1-size)):];
    return train_x,train_y,test_x,test_y; 

