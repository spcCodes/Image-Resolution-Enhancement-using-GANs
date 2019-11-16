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


#importing local packages
from utils import utils as ut

image_shape = (384,384,3)

def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan



def hr_images(images):
    images_hr = np.array(images)
    return images_hr


def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        images.append(
            imresize(images_real[img], [images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale],
                     interp='bicubic', mode=None))
    images_lr = np.array(images)
    return images_lr


def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x, dtype=np.float32)


def deprocess_HR(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    x = np.clip(x * 255, 0, 255)
    return x


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def deprocess_LRS(x):
    x = np.clip(x * 255, 0, 255)
    return x.astype(np.uint8)
