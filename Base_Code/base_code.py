import os
import subprocess

print("Script start")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
result = subprocess.run("/software/gpucheck/gpuuse.sh", shell=True, stdout=subprocess.PIPE)
free_gpu = result.stdout.decode('utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
print()
print("Found a GPU")
print(str(free_gpu))

import numpy as np
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)
print("imported tensorflow") 
#keras = tf.keras
from tensorflow.keras.layers import Conv3D, Dropout, MaxPooling3D, concatenate, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import nibabel as nib
import pickle
import math

mask_path = "/gpfs01/home/bbzfk/CBFPredict_analysis/brain_mask.nii.gz"
mask_obj = nib.load(mask_path)
mask = mask_obj.get_data()
batch_mask = mask #np.stack((mask, mask, mask, mask, mask, mask, mask, mask), axis=0)
print("Mask shape is:")
print(batch_mask.shape)
batch_mask = np.expand_dims(batch_mask, axis=[0,4])
print("Batch mask shape is:")
print(batch_mask.shape)
batch_mask = tf.convert_to_tensor(batch_mask, dtype=tf.float32)

def masked_mse(y_true, y_pred):
    print("Batch size is:")
    print(y_true.shape[0])
    sq = tf.square(y_true - y_pred)
    masked_sq = tf.multiply(sq, batch_mask)
    print("Masked square shape is:")
    print(masked_sq.shape)
    loss = tf.reduce_sum(masked_sq, axis=[1,2,3,4])
    print("Loss shape is:")
    print(loss.shape)
    return loss

def step_decay_ten(epoch):
    step = 20
    initial_power = -4
    # The learning rate begins at 10^initial_power,
    # and decreases by a factor of 10 every step epochs.
    num = epoch // step
    lrate = (10**(initial_power - num))
    print("Learning rate for epoch {} is {}.".format(epoch+1, 1.0*lrate))
    return np.float(lrate)

def step_decay(epoch):
    initial_lrate = 0.00001
    drop = 0.5
    epochs_drop = 25
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("Learning rate for epoch {} is {}.".format(epoch+1, lrate))
    return np.float(lrate)

def unet(input_size = (64,72,64,2)):
    unet_inputs = tf.keras.Input(input_size)#(64,72,64,2))#
    conv1 = Conv3D(32, (3,3,3), activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(unet_inputs)
    conv1 = Conv3D(32, (3,3,3), activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)

    conv2 = Conv3D(32, (3,3,3), activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(64, (3,3,3), activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)

    conv3 = Conv3D(64, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(128, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Conv3D(128, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv3D(128, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop4))
    merge5 = concatenate([drop3,up5], axis = 4)
    conv5 = Conv3D(128, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv3D(128, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv3D(64, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv5))
    merge6 = concatenate([drop2,up6], axis = 4)
    conv6 = Conv3D(64, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(64, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv3D(32, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([drop1,up7], axis = 4)
    conv7 = Conv3D(32, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(32, 3, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv8 = Conv3D(1, 1, activation = 'relu', use_bias=True, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    model = tf.keras.Model(inputs = unet_inputs, outputs = conv8)

    model.compile(optimizer = Adam(1e-5), loss=masked_mse)#, metrics = ['mean_squared_error']) 'mean_squared_error'

    # model.summary()
    return model

model = unet()

callbacks = [LearningRateScheduler(step_decay)]

path_root = "/gpfs01/home/bbzfk/CBFPredict_analysis/grant_train_set/"
subjects = os.listdir(path_root)

input_train_data = np.empty([87, 64, 72, 64, 2])
print(input_train_data.shape)
target_train = np.empty([87, 64, 72, 64])
# print(target_train_data.shape)
iterant = 0
for i in subjects:
    input_path = (path_root + i + "/pad_pves.nii.gz")
    target_path = (path_root + i + "/pad_perfusion.nii.gz")

    input_images = nib.load(input_path)
    input_data = input_images.get_data()
    input_train_data[iterant,:,:,:,:] = input_data

    target_images = nib.load(target_path)
    target_data = target_images.get_data()
    target_train[iterant,:,:,:] = target_data
    target_train_data = np.expand_dims(target_train, axis=4)
    target_train_data[target_train_data > 150] = 150
#    print(target_train_data.shape)

    iterant += 1

target_train_data = target_train_data / 150
print(target_train_data.shape)

history = model.fit(x=input_train_data,
             y=target_train_data,
             batch_size=4,
             epochs=400,
             callbacks=callbacks,
             verbose=2,
             validation_split=0.10,
             )

model.save("/gpfs01/home/bbzfk/CBFPredict_analysis/grant_test.h5")

with open('/gpfs01/home/bbzfk/CBFPredict_analysis/grant_test_trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
