# Import Files 
import os
import re
import nibabel as nib
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt


### This will be the first try on the code

## Cluster Paths
# mri_data_folder = '/gpfs01/share/TILDA/Raw_nii_files_for_MC_pCASL_T1_B0_M0/'
# asl_data_folder = '/gpfs01/share/TILDA/Processed_pCASL/Baseline/'

## Personal Laptop Paths
mri_data_folder = '/Users/anshuman/Desktop/TILDA_DATA/Raw Files/Raw_nii_files_for_MC_pCASL_T1_B0_M0/'
asl_data_folder = '/Users/anshuman/Desktop/TILDA_DATA/Baseline/'

## Lists to store the path of mri data and asl data tensors
mri_data_path = []; asl_data_path = []; 

## Lists to store mri and asl image tensors
mri_images = []; asl_images = [];

### Loading Data 
## Loading  MRI DATA
mri_regex_1 = re.compile(r'__MPR.*') ; mri_regex_2 = re.compile(r'(WIP)*T13D'); 

# Iterating over Directories 
for subdir, dirs, files in os.walk(mri_data_folder):
    for file in files:
        # Storing the path of Data
        if mri_regex_1.search(file) == None:
            if mri_regex_2.search(file): mri_data_path.append(os.path.join(subdir, file));

# Printing the path of MRI DATA
print('\nMRI DATA PATH'); 
for i in mri_data_path: print(i); 

## Loading ASL data
for subdir, dirs, files in os.walk(asl_data_folder):
    # Selecting the ASL file
    for file in files:
        if file == 'asldata.nii': asl_data_path.append(os.path.join(subdir, file));

# Printing the path of ASL images
print('\n\nASL DATA PATH'); 
for i in asl_data_path: print(i);


### Loading the Image Tensor from the paths available in data paths lists
## MRI image tensors 

# Loading data in the image vector
for i in mri_data_path: mri_images.append(nib.load(i)); 


# Printing the shape of images stored and its data type
print('\n\nMRI image tensors'); 
for i in mri_images: print('Image shape ->',i.shape,'\t','Image Data Type ->',i.get_data_dtype());

## ASL image tensors 

# Loading data in the image vector
for i in asl_data_path: asl_images.append(nib.load(i)); 

# Printing the shape of images stored and its data type
print('\n\nASL image tensors');
for i in asl_images: print('Image shape ->',i.shape,'\t','Image Data Type ->',i.get_data_dtype());