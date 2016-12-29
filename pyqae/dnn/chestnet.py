import os
from warnings import warn

warn("ChestNet has been replaced with PETNET and will not be developed in the future", DeprecationWarning)
try:
    assert os.environ['KERAS_BACKEND'] == 'theano', "Theano backend is expected!"
except KeyError:
    print("Backend for keras is undefined setting to theano for CHESTNET")
    os.environ['KERAS_BACKEND'] = 'theano'

from keras import backend as K

K.set_image_dim_ordering('th')

from keras.layers import Cropping2D

from keras.models import Model
from keras.layers import Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import merge, Input
import numpy as np
from pyqae.dnn import fix_name_tf
from pyqae.utils import Tuple, List, Dict, Any, Optional


def build_nd_umodel(in_shape, # type: Tuple[int, int, int]
                    layers,
                    depth,
                    conv_op, pool_op, upscale_op,
                    crop_op,
                    layer_size_fcn=lambda i: 3,
                    pool_size=2,
                    dropout_rate=0.0,
                    last_layer_depth=4,
                    crop_mode=True,
                    use_b_conv=False,
                    use_bn=True):
    border_mode = 'valid' if crop_mode else 'same'

    chan, x_wid, y_wid = in_shape

    im_shape = (1, x_wid, y_wid)
    pos_shape = (chan - 1, x_wid, y_wid)

    input_im = Input(im_shape, name=fix_name_tf('Chest XRay'))
    fmap = conv_op(depth, 3, name='A Prefiltering', border_mode=border_mode)(input_im)
    first_layer = conv_op(depth, 3, name='B Prefiltering', border_mode=border_mode)(fmap)

    last_layer = first_layer

    conv_layers = []
    pool_layers = []

    # save_layer = crop_op(1, 'Cropping')(last_layer) if crop_mode else last_layer
    conv_layers += [last_layer]

    # the image layers
    for ilay in range(layers):

        # only apply BN to the left pathway

        pool_layers += [pool_op(pool_size, name='Downscaling [{}]'.format(ilay))(last_layer)]
        if use_bn:
            last_layer = BatchNormalization(name=fix_name_tf('Batch Normalization [{}]'.format(ilay)))(pool_layers[-1])
        else:
            last_layer = pool_layers[-1]
        lay_depth = depth * np.power(2, ilay + 1)

        conv_layers += [last_layer]

        # double filters

        # use a different width for the last layer (needed if it is 1x1 convolution)
        if ilay == (layers - 1):
            lay_kern_wid = layer_size_fcn(ilay)
        else:
            lay_kern_wid = layer_size_fcn(ilay)

        f_conv_step = conv_op(lay_depth, lay_kern_wid,
                              border_mode=border_mode, name='A Feature Maps [{}]'.format(ilay, lay_depth))(last_layer)
        post_conv_step = conv_op(lay_depth, lay_kern_wid, border_mode=border_mode,
                                 name='B Feature-Maps [{}]'.format(ilay, lay_depth))(f_conv_step)

        last_layer = post_conv_step

    last_img_layer = last_layer
    # remove the last layer
    print('image layers', [i._keras_shape for i in conv_layers])
    rev_layers = list(reversed(list(zip(range(layers + 1), conv_layers[:]))))
    # rev_layers += [(-1,first_layer)]

    # the position layers
    pos_depth = int(depth / 2)

    input_pos = Input(pos_shape, name=fix_name_tf('Mesh Input Channels'))
    fmap = conv_op(pos_depth, 3, name='A Mesh Prefiltering', border_mode=border_mode)(input_pos)
    first_layer = conv_op(depth, 3, name='B Mesh Prefiltering', border_mode=border_mode)(fmap)

    last_layer = first_layer

    conv_layers = []
    pool_layers = []

    # save_layer = crop_op(1, 'Mesh Cropping')(last_layer) if crop_mode else last_layer
    conv_layers += [last_layer]

    for ilay in range(layers):

        # only apply BN to the left pathway

        pool_layers += [pool_op(pool_size, name='Mesh Downscaling [{}]'.format(ilay))(last_layer)]

        last_layer = pool_layers[-1]

        lay_depth = pos_depth * np.power(2, ilay + 1)

        # double filters

        # use a different width for the last layer (needed if it is 1x1 convolution)
        if ilay == (layers - 1):
            lay_kern_wid = layer_size_fcn(ilay)
        else:
            lay_kern_wid = layer_size_fcn(ilay)

        conv_layers += [last_layer]

        f_conv_step = conv_op(lay_depth, lay_kern_wid, border_mode=border_mode,
                              name='A Mesh Feature Maps [{}]'.format(ilay, lay_depth))(last_layer)
        post_conv_step = conv_op(lay_depth, lay_kern_wid, border_mode=border_mode,
                                 name='B Mesh Feature-Maps [{}]'.format(ilay, lay_depth))(f_conv_step)

        last_layer = post_conv_step

    rev_layers_pos = list(reversed(list(zip(range(layers + 1), conv_layers))))

    last_layer = last_img_layer
    last_layer = None

    for (ilay, l_pool), (_, p_pool) in zip(rev_layers, rev_layers_pos):
        print(ilay, l_pool._keras_shape, p_pool._keras_shape)
        lay_depth = depth * np.power(2, ilay)

        merge_layers = [] if last_layer is None else [last_layer]

        if crop_mode and len(merge_layers) > 0:
            crop_d = int((l_pool._keras_shape[2] - last_layer._keras_shape[2]) / 2)
            if crop_d > 0:
                print('Shape Mismatch:', l_pool._keras_shape[2], last_layer._keras_shape[2], 'crop->', crop_d)
                l_pool = crop_op(crop_d, 'Img Cropping [{}]'.format(ilay))(l_pool)
                p_pool = crop_op(crop_d, 'Mesh Cropping [{}]'.format(ilay))(p_pool)
        merge_layers += [l_pool, p_pool]
        # return Model(input=[input_im, input_pos], output=merge_layers)
        print('merging:', [x._keras_shape for x in merge_layers])
        cur_merge = merge(merge_layers, mode='concat',
                          concat_axis=1,
                          name=fix_name_tf('Mixing Original Data [{}]'.format(ilay + 1)))

        last_layer = cur_merge

        if ilay > 0:
            cur_up = upscale_op(pool_size, name='Upsampling [{}]'.format(ilay + 1))(last_layer)
            last_layer = cur_up

            if dropout_rate > 0: cur_merge = Dropout(dropout_rate,
                                                     name=fix_name_tf('Random Removal [{}]'.format(ilay + 1)))(
                cur_merge)
            a_conv = conv_op(lay_depth, 3, name='A Deconvolution [{}]'.format(ilay + 1), border_mode=border_mode)(
                last_layer)
            if use_b_conv:
                b_conv = conv_op(lay_depth, 3, name='B Deconvolution {}]'.format(ilay + 1), border_mode=border_mode)(
                    a_conv)
                last_layer = b_conv
            else:
                last_layer = a_conv

    out_conv = conv_op(last_layer_depth, 1, activation='sigmoid',
                       name='Calculating Probability', border_mode=border_mode)(last_layer)
    model = Model(input=[input_im, input_pos], output=out_conv)

    return model


def build_2d_umodel(in_img, layers,
                    depth=8, lsf=lambda i: 3, pool_size=3,
                    dropout_rate=0, last_layer_depth=1):
    """

    :param in_img:
    :param layers:
    :param depth:
    :param lsf:
    :param pool_size:
    :param dropout_rate:
    :param last_layer_depth:
    :return:

    >>> build_2d_umodel(np.zeros((1 ,3 ,64, 64)), 2, depth = 8, dropout_rate=0.5, last_layer_depth = 4)
    """
    conv_op = lambda n_filters, f_width, name, activation='relu', border_mode='same', **kwargs: Convolution2D(n_filters,
                                                                                                              f_width,
                                                                                                              f_width,
                                                                                                              activation=activation,
                                                                                                              border_mode=border_mode,
                                                                                                              name=fix_name_tf(
                                                                                                                  name),
                                                                                                              **kwargs)
    pool_op = lambda p_size, name, **kwargs: MaxPooling2D(pool_size=(p_size, p_size), name=fix_name_tf(name), **kwargs)
    upscale_op = lambda p_size, name, **kwargs: UpSampling2D(size=(p_size, p_size), name=fix_name_tf(name), **kwargs)
    crop_op = lambda crop_d, name, **kwargs: Cropping2D(cropping=((crop_d, crop_d), (crop_d, crop_d)),
                                                        name=fix_name_tf(name))
    return build_nd_umodel(in_img.shape[1:], layers, depth, conv_op=conv_op, pool_op=pool_op, upscale_op=upscale_op,
                           crop_op=crop_op,
                           layer_size_fcn=lsf, pool_size=pool_size,
                           dropout_rate=dropout_rate,
                           last_layer_depth=last_layer_depth,
                           use_bn=True, crop_mode=True)


if __name__ == "__main__":
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import chestnet

    doctest.testmod(chestnet, verbose=True, report=True)
