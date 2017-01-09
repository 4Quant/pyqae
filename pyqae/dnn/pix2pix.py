from __future__ import print_function, division, absolute_import
import os
try:
    assert os.environ['KERAS_BACKEND'] == 'tensorflow', "Tensorflow backend is expected!"
except KeyError:
    print("Backend for keras is undefined setting to tensorflow for Pix2Pix")
    os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import generic_utils
import keras.backend as K
import numpy as np
# noinspection PyPep8
from pyqae.utils import Tuple, List, Optional

VERBOSE = False
SAVE_FIGURES = False

__doc__ = """
Pix2Pix is a generative model to go from images to other images.
The code is modified from https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix/src/model
And changed so it supports
"""

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]

def _get_conv_size(x):
    if VERBOSE: print('current shape:', x._keras_shape)
    if K.image_dim_ordering() == "tf":
        _, x_wid, y_wid, _ = x._keras_shape
    else:
        _, _, x_wid, y_wid = x._keras_shape
    xw_wid = min(x_wid, 3)
    yw_wid = min(y_wid, 3)
    if VERBOSE: print('conv2d size', xw_wid, yw_wid)
    return xw_wid, yw_wid

def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, subsample=(2,2)):

    x = LeakyReLU(0.2)(x)
    xw_wid, yw_wid = _get_conv_size(x)
    x = Convolution2D(f, xw_wid, yw_wid, subsample=subsample, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    xw_wid, yw_wid = _get_conv_size(x)
    x = Convolution2D(f, xw_wid, yw_wid, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, bn_mode, bn_axis, bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    xw_wid, yw_wid = _get_conv_size(x)
    x = Deconvolution2D(f, xw_wid, yw_wid, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


def generator_unet_upsampling(img_dim, bn_mode, model_name="generator_unet_upsampling"):

    nb_filters = 64

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    if VERBOSE: print(nb_conv, "number of convolutions")
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    xw_wid, yw_wid = _get_conv_size(unet_input)
    list_encoder = [Convolution2D(list_nb_filters[0], xw_wid, yw_wid,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    xw_wid, yw_wid = _get_conv_size(x)
    x = Convolution2D(nb_channels, xw_wid, xw_wid, name="last_conv", border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def generator_unet_deconv(img_dim, bn_mode, batch_size, model_name="generator_unet_deconv"):

    assert K.backend() == "tensorflow", "Not implemented with theano backend"

    nb_filters = 64

    h, w, nb_channels = img_dim
    min_s = min(h, w)
    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    if VERBOSE: print(nb_conv, "number of convolutions")
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    xw_wid, yw_wid = _get_conv_size(unet_input)
    list_encoder = [Convolution2D(list_nb_filters[0], xw_wid, yw_wid,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    # update current "image" h and w
    h, w = h / 2, w / 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-1][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                      list_nb_filters[0], h, w, batch_size,
                                      "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    h, w = h * 2, w * 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                                 w, batch_size, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = Activation("relu")(list_decoder[-1])
    o_shape = (batch_size,) + img_dim
    xw_wid, yw_wid = _get_conv_size(x)
    x = Deconvolution2D(nb_channels, xw_wid, yw_wid, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    xw_wid, yw_wid = _get_conv_size(x_input)
    x = Convolution2D(64, xw_wid, yw_wid, subsample=(2, 2), name="disc_conv2d_1", border_mode="same")(x_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    list_f = [128, 256, 512, 512]
    for i, f in enumerate(list_f):
        name = "disc_conv2d_%s" % (i + 2)
        xw_wid, yw_wid = _get_conv_size(x)
        x = Convolution2D(f, xw_wid, yw_wid, subsample=(2, 2), name=name, border_mode="same")(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    PatchGAN = Model(input=[x_input], output=[x, x_flat], name="PatchGAN")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = merge(x, mode="concat", name="merge_feat")
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = merge(x_mbd, mode="concat", name="merge_feat_mbd")
        else:
            x_mbd = x_mbd[0]

        num_kernels = 100
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        x_mbd = M(x_mbd)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator_model = Model(input=list_input, output=[x_out], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="DCGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "th":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "tf":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)

    DCGAN_model = Model(input=[gen_input],
                  output=[generated_image, DCGAN_output],
                  name="DCGAN")

    return DCGAN_model


def load(model_name, img_dim, nb_patch, bn_mode, use_mbd, batch_size, out_path_fmt = '../../figures/%s.png'):
    """

    :param model_name:
    :param img_dim:
    :param nb_patch:
    :param bn_mode:
    :param use_mbd:
    :param batch_size:
    :param out_path_fmt:
    :return:
    >>> SAVE_FIGURES = True
    >>> VERBOSE = False
    >>> n_model = load("generator_unet_upsampling", (32, 32, 3), 16, 2, False, 32, out_path_fmt='%s.png')
    >>> len(n_model.layers)
    41
    >>> u_model = load("generator_unet_upsampling", (64, 64, 1), 8, 2, False, 32, out_path_fmt='%s_2.png')
    >>> len(u_model.layers)
    49
    >>> u_model.layers[-1].output_shape[1:]
    (64, 64, 1)
    >>> dcg_model = load("DCGAN_discriminator",(64, 64, 1), 8, True, True, False)  # doctest: +ELLIPSIS
    PatchGAN summary
     ...
    discriminator_input (InputLayer) (None, 64, 64, 1)     0
     ...
    Total params: 3916674
     ...
    >>> [(ilay.name, ilay.output_shape[1:]) for ilay in dcg_model.layers]
    [('disc_input_0', (64, 64, 1)), ('disc_input_1', (64, 64, 1)), ('disc_input_2', (64, 64, 1)), ('disc_input_3', (64, 64, 1)), ('disc_input_4', (64, 64, 1)), ('disc_input_5', (64, 64, 1)), ('disc_input_6', (64, 64, 1)), ('disc_input_7', (64, 64, 1)), ('PatchGAN', [(None, 2048)]), ('merge_feat_mbd', (16384,)), ('dense_1', (500,)), ('reshape_1', (100, 5)), ('merge_feat', (16,)), ('lambda_2', (100,)), ('merge_15', (116,)), ('disc_output', (2,))]
    >>> len(dcg_model.layers)
    16
    >>> dcg_model.layers[-1].output_shape[1:]
    (2,)
    >>> # u_model = load("generator_unet_deconv", (256, 256, 3), 16, 2, False, 32, out_path_fmt='%s.png')


    """
    from keras.utils.visualize_util import plot
    if model_name == "generator_unet_upsampling":
        model = generator_unet_upsampling(img_dim, bn_mode, model_name=model_name)
    elif model_name == "generator_unet_deconv":
        model = generator_unet_deconv(img_dim, bn_mode, batch_size, model_name=model_name)
    elif model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
    else:
        raise RuntimeError("The model name is not supported by Pix2Pix currently {}".format(model_name))

    if VERBOSE: print(model.summary())
    if SAVE_FIGURES: plot(model, to_file=out_path_fmt % model_name, show_shapes=True, show_layer_names=True)
    return model

def build_models(
        img_dim, # type: Tuple[int, int, int]
        img_dim_disc, # type: Tuple[int, int, int]
        bn_mode, # type: bool
        use_mbd,
        batch_size, # type: int
        patch_size, # type: Tuple[int, int]
        generator = "upsampling",
        image_dim_ordering = None # type: Optional[str]
    ):
    """

    :param img_dim:
    :param img_dim_disc:
    :param bn_mode:
    :param use_mbd:
    :param batch_size:
    :param patch_size:
    :param generator:
    :param image_dim_ordering:
    :return:
    >>> SAVE_FIGURES = False
    >>> a,b,c = build_models((64, 64, 1), (64, 64, 1), True, False, 8, (64, 64 ))
    """

    assert generator in ["upsampling"], "The only models supported right now are upsampling (deconv doesnt work yet)"
    nb_patch = (img_dim[0]//patch_size[0]) * (img_dim[1]//patch_size[1])

    if image_dim_ordering is None:
        image_dim_ordering = K.image_dim_ordering()

    # Create optimizers
    opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
    opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Load generator model
    generator_model = load("generator_unet_%s" % generator,
                                  img_dim,
                                  nb_patch,
                                  bn_mode,
                                  use_mbd,
                                  batch_size)
    # Load discriminator model
    discriminator_model = load("DCGAN_discriminator",
                                      img_dim_disc,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size)

    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = DCGAN(generator_model,
                               discriminator_model,
                               img_dim,
                               patch_size,
                               image_dim_ordering)

    loss = ['mae', 'binary_crossentropy']
    loss_weights = [1E2, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
    return generator_model, DCGAN_model, discriminator_model


def extract_patches(X, # type: np.ndarray
                    image_dim_ordering, # type: str
                    patch_size # type: Tuple[int, int]
                    ):

    # Now extract patches form X_disc
    if image_dim_ordering == "th":
        # noinspection PyTypeChecker
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] / patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] / patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_dim_ordering == "th":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]

def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_dim_ordering, label_smoothing=False, label_flipping=0, use_patches = False):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    if use_patches:
        X_disc = extract_patches(X_disc, image_dim_ordering, patch_size)

    return X_disc, y_disc

def train_model(generator_model,
                discriminator_model,
                DCGAN_model,
                X_full_train,
                X_sketch_train,
                batch_size,
                nb_epoch,
                epoch_size,
                patch_size,  # type: Tuple[int, int]
                label_smoothing = False,
                label_flipping = False,
                image_dim_ordering=None  # type: Optional[str]
            ):
    gen_loss = 100
    disc_loss = 100

    # Start training
    print("Start training")
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        for X_full_batch, X_sketch_batch in gen_batch(X_full_train, X_sketch_train, batch_size):
            X_disc, y_disc = get_disc_batch(X_full_batch,
                                                       X_sketch_batch,
                                                       generator_model,
                                                       batch_counter,
                                                       patch_size,
                                                       image_dim_ordering,
                                                       label_smoothing=label_smoothing,
                                                       label_flipping=label_flipping)
            # Update the discriminator
            disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
            X_gen_target, X_gen = next(gen_batch(X_full_train, X_sketch_train, batch_size))
            # Create a batch to feed the generator model
            #X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            batch_counter += 1

if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import pix2pix
    doctest.testmod(pix2pix, verbose=True, optionflags=doctest.ELLIPSIS)