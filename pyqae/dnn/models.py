"""
Existing models for Neural Networks
"""
from __future__ import print_function

import numpy as np
from keras.models import Model
import keras.models as models
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def dice_coef(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_unet(img_rows = 64, img_cols = 80, img_channels = 1):
    """
    Create a network modeled after U-NET from Olaf Ronnenberger
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015
    Volume 9351 of the series Lecture Notes in Computer Science pp 234-241
    
    Note: The crop and copy step is currently not implemented
    """
    
    inputs = Input((img_channels, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    return model

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_segnet(img_rows = 360, img_cols = 480, img_channels = 3 , kernel = 3, filter_size = 64, pad = 1, pool_size = 2):
    """
    An implementation of Segnet (http://arxiv.org/pdf/1511.00561v2.pdf)
    By Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla,
    The dimensions optimized to match the VGG networks well and are optimized for street scenes
    """
    def create_encoding_layers():
        return [
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(128, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(256, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(512, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            #MaxPooling2D(pool_size=(pool_size, pool_size)),
        ]

    def create_decoding_layers():
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2
        return[
            #UpSampling2D(size=(pool_size,pool_size)),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(512, kernel, kernel, border_mode='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size,pool_size)),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(256, kernel, kernel, border_mode='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size,pool_size)),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(128, kernel, kernel, border_mode='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size,pool_size)),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
        ]

    autoencoder = models.Sequential()
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
    autoencoder.add(Layer(input_shape=(img_channels, img_rows, img_cols)))

    #autoencoder.add(GaussianNoise(sigma=0.3))
    autoencoder.encoding_layers = create_encoding_layers()
    autoencoder.decoding_layers = create_decoding_layers()
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Convolution2D(12, 1, 1, border_mode='valid',))
    autoencoder.add(Reshape((12,img_rows*img_cols), input_shape=(12,img_rows,img_cols)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    return autoencoder	
