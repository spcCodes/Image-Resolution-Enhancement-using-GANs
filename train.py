#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on NOv 14 , 2019

@author: suman.choudhury
"""

#importing the necessary packages
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize
import os
import random

#importing local packages
from utils.utils import *
from src.super_resolution_gan import Generator,Discriminator
from network_param import *

np.random.seed(10)
#defining an image size for resizing
image_shape = (384,384,3)
image_shape_tuple =(384,384)

#the directory where images are kept
directory_name = "data/"

files = load_data(directory_name, ".jpg",image_shape)
random.seed(42)
random.shuffle(files)

net_images = len(files)
train_test_ratio = 0.8
num_train = int(0.8 * net_images)
num_test = net_images - num_train

x_train = files[:num_train]
x_test = files[num_train:]

print("data has been loaded")

#processing the data and converting the high resolution and low resolution images

x_train_hr = hr_images(x_train)
x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, 4)
x_train_lr = normalize(x_train_lr)


x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
x_test_lr = normalize(x_test_lr)

print("data has been processed")


#defining a training function
def train_gan(epochs=1, batch_size=128):
    downscale_factor = 4

    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])

    generator = Generator(shape).build_generator()()
    discriminator = Discriminator(image_shape).build_discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam)

    for epoch in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % epoch, '-' * 15)
        for i in range(batch_count):

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)

        if epoch % 300 == 0:
            generator.save('models/gen_model%d.h5' % epoch)
            discriminator.save('models/dis_model%d.h5' % epoch)
            gan.save('models/gan_model%d.h5' % epoch)

if __name__== "__main__":

    #training the gan FOR 20000 epochs with batch size of 4
    train_gan(20000, 4)



















