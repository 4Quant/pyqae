from __future__ import print_function, division, absolute_import

__doc__ = """
DeepGenNet is a generative model going from a single input vector to an output image by progressive
UpSampling, Convolution and Concatenations
"""

import os

try:
    assert os.environ['KERAS_BACKEND'] == 'theano', "Theano backend is expected!"
except KeyError:
    print("Backend for keras is undefined setting to theano for DeepGenNet")
    os.environ['KERAS_BACKEND'] = 'theano'

from keras import backend as K

K.set_image_dim_ordering('th')

from keras.models import Model
from keras.layers import Convolution2D, UpSampling2D
from keras.layers import merge, Input, Reshape, Dense, Dropout
from pyqae.dnn import fix_name_tf

def create_tweennet(in_length = 128,
                    layers = 4,
                    f_size = (1, 128, 128),
                    dropout_rate = 0.5,
                    us_size=(2, 2),
                    **kwargs
                    ):
    """

    :param in_length:
    :param layers:
    :param f_size:
    :return:
    >>> simple_mod = create_tweennet(1024, 4, (3, 128, 128))
    >>> len(simple_mod.layers)
    29
    >>> [(x.name, x.output_shape[1:]) for x in simple_mod.layers if x.name.find('Reshape')>=0]
    [('Reshape-0', (16, 8, 8)), ('Reshape-1', (4, 16, 16)), ('Reshape-2', (1, 32, 32)), ('Reshape-3', (1, 64, 64)), ('Reshape-4', (1, 128, 128))]
    >>> simple_mod.layers[-1].output_shape[1:] # check output
    (3, 128, 128)
    >>> [(x.name, x.output_shape[1:]) for x in simple_mod.layers if x.name.find('Mixing')>=0]
    [('Mixing-1', (20, 16, 16)), ('Mixing-2', (17, 32, 32)), ('Mixing-3', (5, 64, 64)), ('Mixing-4', (2, 128, 128))]
    >>> [(x.name, x.output_shape[1:]) for x in simple_mod.layers if x.name.find('Convolution')>=0]
    [('Convolution-1', (16, 16, 16)), ('Convolution-2', (4, 32, 32)), ('Convolution-3', (1, 64, 64)), ('Convolution-4', (1, 128, 128))]
    >>> big_mod = create_tweennet(384, 2, (1, 256, 128))
    >>> len(big_mod.layers)
    17
    >>> [(x.name, x.output_shape[1:]) for x in big_mod.layers if x.name.find('Reshape')>=0]
    [('Reshape-0', (1, 64, 32)), ('Reshape-1', (1, 128, 64)), ('Reshape-2', (1, 256, 128))]
    >>> big_mod.layers[-1].output_shape[1:] # check output
    (1, 256, 128)
    """
    out_depth, f_xwid, f_ywid = f_size
    starting_size = f_xwid // (us_size[0]**layers), f_ywid // (us_size[1]**layers)
    s_depths = [in_length // (starting_size[0]*(us_size[0]**i)*starting_size[1]*(us_size[1]**i)) for i in range(layers+1)]
    s_depths = [max(x,1) for x in s_depths]
    return build_dgnet_2d(in_size = in_length,
                          s_depths = s_depths,
                         starting_size=starting_size,
                          out_depth = out_depth,
                          us_size = us_size,
                          dropout_rate = dropout_rate,
                          **kwargs
                          )



def build_dgnet_2d(in_size,
                   s_depths,
                   conv_depths=None,
                   us_size=(2, 2),
                   starting_size=(2, 2),
                   activation='relu',
                   cnn_size=(3, 3),
                   out_depth=1,
                   dropout_rate = 0):
    """
    Create the DeepGenerative Network
    :param in_size:
    :param s_depths:
    :param conv_depths:
    :param us_size:
    :param starting_size:
    :param activation:
    :param cnn_size:
    :param out_depth:
    :param dropout_rate: use dropout (rate >0)
    :return:
    >>> simple_mod = build_dgnet_2d(128, [4, 2])
    >>> len(simple_mod.layers)
    9
    >>> simple_mod.layers[-1].output_shape[1:]
    (1, 4, 4)
    >>> [(x.name, x.output_shape[1:]) for x in simple_mod.layers]
    [('Input', (128,)), ('Fully-Connected-0', (16,)), ('Reshape-0', (4, 2, 2)), ('Fully-Connected-1', (32,)), ('Upsampling-1', (4, 4, 4)), ('Reshape-1', (2, 4, 4)), ('Mixing-1', (6, 4, 4)), ('Convolution-1', (4, 4, 4)), ('Normalizing-Output', (1, 4, 4))]
    >>> deep_mod = build_dgnet_2d(128, [4, 2, 1], conv_depths = [6, 4, 2])
    >>> len(deep_mod.layers)
    14
    >>> deep_mod.layers[-1].output_shape[1:]
    (1, 8, 8)
    >>> [(x.name, x.output_shape[1:]) for x in deep_mod.layers if x.name.find('Mixing')>=0]
    [('Mixing-1', (6, 4, 4)), ('Mixing-2', (5, 8, 8))]
    >>> [(x.name, x.output_shape[1:]) for x in deep_mod.layers if x.name.find('Convolution')>=0]
    [('Convolution-1', (4, 4, 4)), ('Convolution-2', (2, 8, 8))]
    >>> full_mod = build_dgnet_2d(256, [4, 2, 1], dropout_rate=0.5)
    >>> len(full_mod.layers)
    17
    >>> full_mod.layers[-1].output_shape[1:]
    (1, 8, 8)
    >>> [(x.name, x.output_shape[1:]) for x in full_mod.layers if x.name.find('Fully')>=0]
    [('Fully-Connected-0', (16,)), ('Fully-Connected-1', (32,)), ('Fully-Connected-2', (64,))]
    """
    if conv_depths is None:
        s_depths = list(s_depths)
        conv_depths = [s_depths[0]] + s_depths[:-1] # shift by one
    s_node = Input(shape=(in_size,), name = fix_name_tf('Input'))
    last_layer = None
    for i, (i_depth, c_depth) in enumerate(zip(s_depths, conv_depths)):
        xwid, ywid = starting_size[0] * (us_size[0] ** i), starting_size[1] * (us_size[1] ** i)
        d_node = Dense(xwid * ywid * i_depth,
                       activation=activation,
                       name=fix_name_tf('Fully Connected [{}]'.format(i))
                       )(s_node)

        if dropout_rate>0:
            d_node = Dropout(dropout_rate, name= fix_name_tf('Dropout [{}]'.format(i)))(d_node)

        r_node = Reshape(target_shape=(i_depth, xwid, ywid),
                         name=fix_name_tf('Reshape [{}]'.format(i)))(d_node)
        if last_layer is not None:
            u_node = UpSampling2D(size=us_size, name=fix_name_tf('Upsampling [{}]'.format(i)))(last_layer)
            m_node = merge([u_node, r_node],
                           mode='concat',
                           concat_axis=1,
                           name=fix_name_tf('Mixing [{}]'.format(i)))
            r_node = Convolution2D(c_depth,
                                   cnn_size[0], cnn_size[1],
                                   border_mode='same',
                                   activation=activation,
                                   name=fix_name_tf('Convolution [{}]'.format(i)))(m_node)
        last_layer = r_node
    last_layer = Convolution2D(out_depth,
                               1, 1,  # like a FCL
                               border_mode='same',
                               activation='sigmoid',
                               name=fix_name_tf('Normalizing Output'))(last_layer)
    return Model(input=[s_node], output=last_layer)


if __name__ == "__main__":
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import deepgennet

    doctest.testmod(deepgennet, verbose=True)
