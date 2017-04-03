__doc__ = """The model definitions for the pix2pix network taken from the
retina repository at https://github.com/costapt/vess2ret
"""
import os

import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from pyqae.dnn import KERAS_2

try:
    # keras 2 imports
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.merge import Concatenate
except ImportError:
    print("Keras 2 layers could not be imported defaulting to keras1")
    KERAS_2 = False

K.set_image_dim_ordering('th')


def concatenate_layers(inputs, concat_axis, mode='concat'):
    if KERAS_2:
        assert mode == 'concat', "Only concatenation is supported in this wrapper"
        return Concatenate(axis=concat_axis)(inputs)
    else:
        return merge(inputs=inputs, concat_axis=concat_axis)


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    if KERAS_2:
        return Convolution2D(f,
                             kernel_size=(k, k),
                             padding=border_mode,
                             strides=(s, s),
                             **kwargs)
    else:
        return Convolution2D(f, k, k, border_mode=border_mode,
                             subsample=(s, s),
                             **kwargs)


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    if KERAS_2:
        return Conv2DTranspose(f,
                               kernel_size=(k, k),
                               strides=(s, s),
                               data_format=K.image_data_format(),
                               **kwargs)
    else:
        return Deconvolution2D(f, k, k,
                               output_shape=output_shape,
                               subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    if KERAS_2:
        return BatchNormalization(axis=axis, **kwargs)
    else:
        return BatchNormalization(mode=2,axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, batch_size=1, is_binary=False, name='unet'):
    # type: (int, int, int, int, bool, str) -> keras.models.Model
    """Define a U-Net.

    Input has shape in_ch x 512 x 512
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    activation function, otherwise, a tanh is used.
    >>> K.set_image_dim_ordering('th')
    >>> K.image_data_format()
    'channels_first'
    >>> unet = g_unet(1, 2, 3, batch_size=5, is_binary=True)
    TheanoShapedU-NET
    >>> for ilay in unet.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> unet.summary()  #doctest: +NORMALIZE_WHITESPACE
     _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 1, 512, 512)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 3, 256, 256)       30
    _________________________________________________________________
    batch_normalization (BatchNo (None, 3, 256, 256)       12
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 3, 256, 256)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 6, 128, 128)       168
    _________________________________________________________________
    batch_normalization (BatchNo (None, 6, 128, 128)       24
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 6, 128, 128)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 12, 64, 64)        660
    _________________________________________________________________
    batch_normalization (BatchNo (None, 12, 64, 64)        48
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 12, 64, 64)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 32, 32)        2616
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 32, 32)        96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 32, 32)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 16, 16)        5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 16, 16)        96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 16, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 8, 8)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 8, 8)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 8, 8)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 4, 4)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 4, 4)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 4, 4)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 2, 2)          5208
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 2, 2)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 2, 2)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 24, 1, 1)          2328
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 1, 1)          96
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 1, 1)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 2, 2)          2328
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 2, 2)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 2, 2)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 2, 2)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 2, 2)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 4, 4)          4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 4, 4)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 4, 4)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 4, 4)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 4, 4)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 8, 8)          4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 8, 8)          96
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 8, 8)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 8, 8)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 8, 8)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 16, 16)        4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 16, 16)        96
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 16, 16)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 16, 16)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 24, 32, 32)        4632
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 32, 32)        96
    _________________________________________________________________
    concatenate (Concatenate)    (None, 48, 32, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 48, 32, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 12, 64, 64)        2316
    _________________________________________________________________
    batch_normalization (BatchNo (None, 12, 64, 64)        48
    _________________________________________________________________
    concatenate (Concatenate)    (None, 24, 64, 64)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 24, 64, 64)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 6, 128, 128)       582
    _________________________________________________________________
    batch_normalization (BatchNo (None, 6, 128, 128)       24
    _________________________________________________________________
    concatenate (Concatenate)    (None, 12, 128, 128)      0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 12, 128, 128)      0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 3, 256, 256)       147
    _________________________________________________________________
    batch_normalization (BatchNo (None, 3, 256, 256)       12
    _________________________________________________________________
    concatenate (Concatenate)    (None, 6, 256, 256)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 6, 256, 256)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 2, 512, 512)       50
    _________________________________________________________________
    activation (Activation)      (None, 2, 512, 512)       0
    =================================================================
    Total params: 51,809.0
    Trainable params: 51,197.0
    Non-trainable params: 612.0
    _________________________________________________________________
    >>> K.set_image_dim_ordering('tf')
    >>> K.image_data_format()
    'channels_last'
    >>> unet2=g_unet(3, 4, 2, batch_size=7, is_binary=False)
    TensorflowShapedU-NET
    >>> for ilay in unet2.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> unet2.summary()  #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 512, 512, 3)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 256, 256, 2)       56
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 2)       1024
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 256, 256, 2)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 128, 128, 4)       76
    _________________________________________________________________
    batch_normalization (BatchNo (None, 128, 128, 4)       512
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 128, 128, 4)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 8)         296
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 8)         256
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 64, 64, 8)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 32, 32, 16)        1168
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 16)        128
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 32, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 16, 16, 16)        64
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 16, 16)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 8, 8, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 16)          32
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 8, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 4, 4, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 4, 4, 16)          16
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 4, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 2, 2, 16)          2320
    _________________________________________________________________
    batch_normalization (BatchNo (None, 2, 2, 16)          8
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 2, 16)          0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 1, 1, 16)          1040
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1, 1, 16)          4
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 1, 1, 16)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 2, 2, 16)          1040
    _________________________________________________________________
    batch_normalization (BatchNo (None, 2, 2, 16)          8
    _________________________________________________________________
    dropout (Dropout)            (None, 2, 2, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 2, 2, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 2, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 4, 4, 16)          2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 4, 4, 16)          16
    _________________________________________________________________
    dropout (Dropout)            (None, 4, 4, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 4, 4, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 4, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 8, 8, 16)          2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 16)          32
    _________________________________________________________________
    dropout (Dropout)            (None, 8, 8, 16)          0
    _________________________________________________________________
    concatenate (Concatenate)    (None, 8, 8, 32)          0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 8, 32)          0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 16, 16, 16)        2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 16, 16, 16)        64
    _________________________________________________________________
    concatenate (Concatenate)    (None, 16, 16, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 16, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 32, 32, 16)        2064
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 16)        128
    _________________________________________________________________
    concatenate (Concatenate)    (None, 32, 32, 32)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 32, 32)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 64, 64, 8)         1032
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 8)         256
    _________________________________________________________________
    concatenate (Concatenate)    (None, 64, 64, 16)        0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 64, 64, 16)        0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 128, 128, 4)       260
    _________________________________________________________________
    batch_normalization (BatchNo (None, 128, 128, 4)       512
    _________________________________________________________________
    concatenate (Concatenate)    (None, 128, 128, 8)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 128, 128, 8)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 256, 256, 2)       66
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 2)       1024
    _________________________________________________________________
    concatenate (Concatenate)    (None, 256, 256, 4)       0
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 256, 256, 4)       0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 512, 512, 4)       68
    _________________________________________________________________
    activation (Activation)      (None, 512, 512, 4)       0
    =================================================================
    Total params: 26,722.0
    Trainable params: 24,680.0
    Non-trainable params: 2,042.0
    _________________________________________________________________
    """
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }
    if True:
        if K.image_dim_ordering() == 'th':
            print('TheanoShapedU-NET')
            i = Input(shape=(in_ch, 512, 512))

            def get_deconv_shape(samples, channels, x_dim, y_dim):
                return samples, channels, x_dim, y_dim

        elif K.image_dim_ordering() == 'tf':
            i = Input(shape=(512, 512, in_ch))
            print('TensorflowShapedU-NET')

            def get_deconv_shape(samples, channels, x_dim, y_dim):
                return samples, x_dim, y_dim, channels

            merge_params['concat_axis'] = 3
        else:
            raise ValueError(
                'Keras dimension ordering not supported: {}'.format(
                    K.image_dim_ordering()))
    else:
        i = Input(shape=(in_ch, 512, 512))
    # in_ch x 512 x 512
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 4 x 4

    conv8 = Convolution(nf * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 2 x 2

    conv9 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 2, 2),
                           k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    try:
        x = concatenate_layers([dconv1, conv8], **merge_params)
    except ValueError:
        return Model(i, dconv1, name=name)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = concatenate_layers([dconv2, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = concatenate_layers([dconv3, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4,
                           get_deconv_shape(batch_size, nf * 4, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = concatenate_layers([dconv6, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2,
                           get_deconv_shape(batch_size, nf * 2, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = concatenate_layers([dconv7, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf,
                           get_deconv_shape(batch_size, nf, 256, 256))(x)
    dconv8 = BatchNorm()(dconv8)
    x = concatenate_layers([dconv8, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(out_ch,
                           get_deconv_shape(batch_size, out_ch, 512, 512))(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(i, out, name=name)

    return unet


def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.

    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    >>> K.set_image_dim_ordering('th')
    >>> disc=discriminator(3,4,2)
    >>> for ilay in disc.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> disc.summary() #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 7, 512, 512)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 2, 256, 256)       128
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 2, 256, 256)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 4, 128, 128)       76
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 4, 128, 128)       0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 8, 64, 64)         296
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8, 64, 64)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 32, 32)        1168
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 16, 32, 32)        0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 1, 16, 16)         145
    _________________________________________________________________
    activation (Activation)      (None, 1, 16, 16)         0
    =================================================================
    Total params: 1,813.0
    Trainable params: 1,813.0
    Non-trainable params: 0.0
    _________________________________________________________________
    """
    i = Input(shape=(a_ch + b_ch, 512, 512))

    # (a_ch + b_ch) x 512 x 512
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(1)(x)
    out = Activation('sigmoid')(conv5)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    # type: (...) -> keras.models.Model
    """
    Define the pix2pix network.
    :param atob:
    :param d:
    :param a_ch:
    :param b_ch:
    :param alpha:
    :param is_a_binary:
    :param is_b_binary:
    :param opt:
    :param name:
    :return:
    >>> K.set_image_dim_ordering('th')
    >>> unet = g_unet(3, 4, 2, batch_size=8, is_binary=False)
    TheanoShapedU-NET
    >>> disc=discriminator(3,4,2)
    >>> pp_net=pix2pix(unet, disc, 3, 4)
    >>> for ilay in pp_net.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    >>> pp_net.summary()  #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 3, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 4, 512, 512)       23454
    _________________________________________________________________
    concatenate (Concatenate)    (None, 7, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 1, 16, 16)         1813
    =================================================================
    Total params: 25,267.0
    Trainable params: 24,859.0
    Non-trainable params: 408.0
    _________________________________________________________________
    """
    a = Input(shape=(a_ch, 512, 512))
    b = Input(shape=(b_ch, 512, 512))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate_layers([a, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + alpha * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix


## Train functions
import numpy as np
class DirectDict(dict):
    """
    Dictionary that allows to access elements with dot notation.
    ex:
    >>> d = DirectDict({'key': 'val'})
    >>> d.key
    'val'
    >>> d.key2 = 'val2'
    >>> sorted(d.items(),key=lambda x: x[0])
    [('key', 'val'), ('key2', 'val2')]
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def discriminator_generator(it, atob, dout_size):
    """
    Generate batches for the discriminator.
    Parameters:
    - it: an iterator that returns a pair of images;
    - atob: the generator network that maps an image to another representation;
    - dout_size: the size of the output of the discriminator.
    """
    while True:
        # Fake pair
        a_fake, _ = next(it)
        b_fake = atob.predict(a_fake)

        # Real pair
        a_real, b_real = next(it)

        # Concatenate the channels. Images become (ch_a + ch_b) x 256 x 256
        fake = np.concatenate((a_fake, b_fake), axis=1)
        real = np.concatenate((a_real, b_real), axis=1)

        # Concatenate fake and real pairs into a single batch
        batch_x = np.concatenate((fake, real), axis=0)

        # 1 is fake, 0 is real
        batch_y = np.ones((batch_x.shape[0], 1) + dout_size)
        batch_y[fake.shape[0]:] = 0

        yield batch_x, batch_y


def train_discriminator(d, it, samples_per_batch=20):
    """Train the discriminator network."""
    return d.fit_generator(it, samples_per_epoch=samples_per_batch*2, nb_epoch=1, verbose=False)


def pix2pix_generator(it, dout_size):
    """
    Generate data for the generator network.
    Parameters:
    - it: an iterator that returns a pair of images;
    - dout_size: the size of the output of the discriminator.
    """
    for a, b in it:
        # 1 is fake, 0 is real
        y = np.zeros((a.shape[0], 1) + dout_size)
        yield [a, b], y


def train_pix2pix(pix2pix, it, samples_per_batch=20):
    """Train the generator network."""
    return pix2pix.fit_generator(it, nb_epoch=1, samples_per_epoch=samples_per_batch, verbose=False)


def evaluate(models, generators, losses, val_samples=192):
    """Evaluate and display the losses of the models."""
    # Get necessary generators
    d_gen = generators.d_gen_val
    p2p_gen = generators.p2p_gen_val

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Evaluate
    d_loss = d.evaluate_generator(d_gen, val_samples)
    p2p_loss = p2p.evaluate_generator(p2p_gen, val_samples)

    losses['d_val'].append(d_loss)
    losses['p2p_val'].append(p2p_loss)

    print('')
    print ('Train Losses of (D={0} / P2P={1});\n'
           'Validation Losses of (D={2} / P2P={3})'.format(
                losses['d'][-1], losses['p2p'][-1], d_loss, p2p_loss))

    return d_loss, p2p_loss


def model_creation(d, atob, params):
    """Create all the necessary models."""
    opt = Adam(lr=params.lr, beta_1=params.beta_1)
    p2p = pix2pix(atob, d, params.a_ch, params.b_ch, alpha=params.alpha, opt=opt,
                    is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary)

    models = DirectDict({
        'atob': atob,
        'd': d,
        'p2p': p2p,
    })

    return models


def generators_creation(it_train, it_val, models, dout_size):
    """Create all the necessary data generators."""
    # Discriminator data generators
    d_gen = discriminator_generator(it_train, models.atob, dout_size)
    d_gen_val = discriminator_generator(it_val, models.atob, dout_size)

    # Workaround to make tensorflow work. When atob.predict is called the first
    # time it calls tf.get_default_graph. This should be done on the main thread
    # and not inside fit_generator. See https://github.com/fchollet/keras/issues/2397
    next(d_gen)

    # pix2pix data generators
    p2p_gen = pix2pix_generator(it_train, dout_size)
    p2p_gen_val = pix2pix_generator(it_val, dout_size)

    generators = DirectDict({
        'd_gen': d_gen,
        'd_gen_val': d_gen_val,
        'p2p_gen': p2p_gen,
        'p2p_gen_val': p2p_gen_val,
    })

    return generators


def train_iteration(models, generators, losses, params):
    """Perform a train iteration."""
    # Get necessary generators
    d_gen = generators.d_gen
    p2p_gen = generators.p2p_gen

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Update the dscriminator
    dhist = train_discriminator(d, d_gen, samples_per_batch=params.samples_per_batch)
    losses['d'].extend(dhist.history['loss'])

    # Update the generator
    p2phist = train_pix2pix(p2p, p2p_gen, samples_per_batch=params.samples_per_batch)
    losses['p2p'].extend(p2phist.history['loss'])


from keras.preprocessing.image import apply_transform, flip_axis
from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import Iterator, load_img, img_to_array


class TwoArrayIterator(Iterator):
    """Class to iterate A and B images at the same time.
    Examples
    --------
    >>> im_a_shape, im_b_shape=(5,3,2,2), (5,7,2,2)
    >>> ti_train=TwoArrayIterator(np.zeros(im_a_shape),np.ones(im_b_shape), target_size=im_a_shape[2:], batch_size=4, dim_ordering='th', seed=1234)
    >>> batch_a, batch_b = next(ti_train)
    >>> batch_a.shape
    (4, 3, 2, 2)
    >>> batch_b.shape
    (4, 7, 2, 2)
    >>> ['%2.2f' % np.std(c_img) for c_img in batch_a]
     ['0.00', '0.00', '0.00', '0.00']
    """

    def __init__(self,
                 a_np_arr,
                 b_np_arr,
                 is_a_binary=False, is_b_binary=False,
                 target_size=(256, 256), rotation_range=0.,
                 height_shift_range=0., width_shift_range=0., zoom_range=0.,
                 fill_mode='constant', cval=0., horizontal_flip=False,
                 vertical_flip=False,  dim_ordering='default', N=-1,
                 batch_size=32, shuffle=True, seed=None):
        """
        Iterate through two directories at the same time.
        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - is_a_binary: converts A images to binary images. Applies a threshold of 0.5.
        - is_b_binary: converts B images to binary images. Applies a threshold of 0.5.
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator.
        """

        self.filenames=range(a_np_arr.shape[0])

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            np.random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise Exception("""Did not find any pair in the dataset. Please check that """
                            """the names and extensions of the pairs are exactly the same. """
                            """Searched inside folders: {0} and {1}""")

        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ('th', 'default', 'tf'):
            raise Exception('dim_ordering should be one of "th", "tf" or "default". '
                            'Got {0}'.format(self.dim_ordering))

        self.target_size = target_size

        self.is_a_binary = is_a_binary
        self.is_b_binary = is_b_binary



        self.load_to_memory = True

        self.a = a_np_arr
        self.b = b_np_arr



        if self.dim_ordering in ('th', 'default'):
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        self.image_shape_a = self._get_image_shape(a_np_arr.shape[
                                                       self.channel_index])
        self.image_shape_b = self._get_image_shape(
                                                   b_np_arr.shape[
                                                       self.channel_index]
                                                   )

        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        super(TwoArrayIterator, self).__init__(len(self.filenames), batch_size,
                                               shuffle, seed)

    def _get_image_shape(self, channel_count):
        """Auxiliar method to get the image shape given the color mode."""
        if self.dim_ordering == 'tf':
            return self.target_size + (channel_count,)
        else:
            return (channel_count,) + self.target_size

    def _binarize(self, batch):
        """Make input binary images have 0 and 1 values only."""
        bin_batch = batch / 255.
        bin_batch[bin_batch >= 0.5] = 1
        bin_batch[bin_batch < 0.5] = 0
        return bin_batch

    def _normalize_for_tanh(self, batch):
        """Make input image values lie between -1 and 1."""
        tanh_batch = batch - 127.5
        tanh_batch /= 127.5
        return tanh_batch

    def _load_img_pair(self, idx, load_from_memory):
        """Get a pair of images with index idx."""
        if load_from_memory:
            a = self.a[idx]
            b = self.b[idx]
            return a, b

        fname = self.filenames[idx]

        a = load_img(os.path.join(self.a_dir, fname),
                     grayscale=self.is_a_grayscale,
                     target_size=self.target_size)
        b = load_img(os.path.join(self.b_dir, fname),
                     grayscale=self.is_b_grayscale,
                     target_size=self.target_size)

        a = img_to_array(a, self.dim_ordering)
        b = img_to_array(b, self.dim_ordering)

        return a, b

    def _random_transform(self, a, b):
        """
        Random dataset augmentation.
        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """
        # a and b are single images, so they don't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)

        h, w = a.shape[img_row_index], a.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        a = apply_transform(a, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        b = apply_transform(b, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_col_index)
                b = flip_axis(b, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_row_index)
                b = flip_axis(b, img_row_index)

        return a, b

    def next(self):
        """Get the next pair of the sequence."""
        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        batch_a = np.zeros((current_batch_size,) + self.image_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.image_shape_b)

        for i, j in enumerate(index_array):
            a_img, b_img = self._load_img_pair(j, self.load_to_memory)
            a_img, b_img = self._random_transform(a_img, b_img)

            batch_a[i] = a_img
            batch_b[i] = b_img

        if self.is_a_binary:
            batch_a = self._binarize(batch_a)
        else:
            batch_a = self._normalize_for_tanh(batch_a)

        if self.is_b_binary:
            batch_b = self._binarize(batch_b)
        else:
            batch_b = self._normalize_for_tanh(batch_b)

        return [batch_a, batch_b]

if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import pix2pix

    TEST_TF = True
    if TEST_TF:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    else:
        os.environ['KERAS_BACKEND'] = 'theano'
    doctest.testmod(pix2pix, verbose=True, optionflags=doctest.ELLIPSIS)
