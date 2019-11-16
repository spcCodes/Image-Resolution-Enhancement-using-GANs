#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on NOv 14 , 2019

@author: suman.choudhury
"""

"""
Since GAN contains 3 networks generator, discriminator and  a pretrained models , we will write the implementation out here 

"""

#importing the necessary modules

from utils import utils as ut
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

# Defining the network Architecture

class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def build_generator(self):

        # Input Layer of the generator network
        gen_input = Input(shape=self.noise_shape)

        # Add the pre-residual block
        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        gen_model = model

        # Using 16 Residual Blocks, thus using a for loop to generate 16 residual blocks
        for index in range(16):
            model = ut.residual_block_generator(model, 3, 64, 1)


        # Adding the post-residual block
        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        # Using 2 UpSampling Blocks
        for index in range(2):
            model = ut.up_sampling(model, 3, 256, 1)

        # Output convolution layer
        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)

        # Keras model
        generator_model = Model(inputs=gen_input, outputs=model)

        return generator_model

class Discriminator(object):

    def __init__(self, image_shape):

        self.image_shape = image_shape

    def build_discriminator(self):


        # Input Layer of the discriminator network
        dis_input = Input(shape=self.image_shape)

        # Add the first convolution block
        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        # Add the second convolution block
        model = ut.discriminator_block(model, 64, 3, 2)

        # Add the third convolution bloc
        model = ut.discriminator_block(model, 128, 3, 1)
        # Add the fourth convolution bloc
        model = ut.discriminator_block(model, 128, 3, 2)
        # Add the fifth convolution bloc
        model = ut.discriminator_block(model, 256, 3, 1)
        # Add the sixth convolution bloc
        model = ut.discriminator_block(model, 256, 3, 2)
        # Add the seventh convolution bloc
        model = ut.discriminator_block(model, 512, 3, 1)
        # Add the eight convolution bloc
        model = ut.discriminator_block(model, 512, 3, 2)

        #flattening out the outputs of the convolution block
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        #adding the final output layer
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model


























