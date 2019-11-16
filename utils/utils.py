#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on NOv 14 , 2019

@author: suman.choudhury
"""


"""
This function will contain all the utility required by the entire project . This may contain network functions or data-preprocessing functions

"""


#import necessary modules

# Modules
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import os
import cv2
import numpy as np
import skimage.transform
from skimage import data, io, filters
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize


# Residual block
def residual_block_generator(model, kernel_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model

#defining an upsampling block after the residual block
def up_sampling(model, kernel_size, filters, strides):

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model

#defining a discriminator block
def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model

#loading the path
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories

#loading the data from directory
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = cv2.imread(os.path.join(d, f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d, f))
                count = count + 1
    return files

#loading the data from the directory and resizing it
def load_data_from_dirs_resize(dirs, ext, size):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                files.append(cv2.resize(cv2.imread(os.path.join(d, f)), size))
                file_names.append(os.path.join(d, f))
                count = count + 1
    return files

#loading the data
def load_data(directory, ext, image_shape):

    files = load_data_from_dirs_resize(load_path(directory), ext, image_shape)
    #files = load_data_from_dirs(load_path(directory), ext)
    return files

