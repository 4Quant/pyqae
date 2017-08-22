from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, \
    concatenate, Dropout, GlobalAveragePooling2D, \
    warnings, BatchNormalization
from keras.models import Model
from keras.utils import get_file
from keras.utils import layer_utils

__doc__ = """
SqueezeNet and the pretrained weights for simple/fast classification problems
"""
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
# a pretrained imagenet
WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"


# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    """
    The fire module component of the SqueezeNet architecture
    :param x:
    :param fire_id:
    :param squeeze:
    :param expand:
    :return:
    >>> in_node = Input(shape = (64, 64, 62), name = "Start")
    >>> out_node = fire_module(in_node, 1, 16, 17)
    >>> Model(inputs=[in_node], outputs = [out_node]).summary()  # doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    Start (InputLayer)               (None, 64, 64, 62)    0
    ____________________________________________________________________________________________________
    fire1/squeeze1x1 (Conv2D)        (None, 64, 64, 16)    1008        Start[0][0]
    ____________________________________________________________________________________________________
    fire1/relu_squeeze1x1 (Activatio (None, 64, 64, 16)    0           fire1/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire1/expand1x1 (Conv2D)         (None, 64, 64, 17)    289         fire1/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire1/expand3x3 (Conv2D)         (None, 64, 64, 17)    2465        fire1/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire1/relu_expand1x1 (Activation (None, 64, 64, 17)    0           fire1/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire1/relu_expand3x3 (Activation (None, 64, 64, 17)    0           fire1/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire1/concat (Concatenate)       (None, 64, 64, 34)    0           fire1/relu_expand1x1[0][0]
                                                                       fire1/relu_expand3x3[0][0]
    ====================================================================================================
    Total params: 3,762
    Trainable params: 3,762
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    """
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(
        x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(
        x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(input_tensor=None, input_shape=None,
               weights='imagenet',
               classes=1000,
               use_bn_on_input=False,  # to avoid preprocessing
               first_stride=2,
               last_activation = 'softmax',
                load_by_name = False
               ):
    """
    The implementation of SqueezeNet in Keras
    :param input_tensor:
    :param input_shape:
    :param weights:
    :param classes:
    :param use_bn_on_input:
    :param first_stride:
    :param last_activation: the activation of the last layer
    :param load_by_name: load the layers by name
    :return:
    >>> m = SqueezeNet(input_shape = (48, 48, 3), weights = 'imagenet')
    >>> m.summary() # doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    input_1 (InputLayer)             (None, 48, 48, 3)     0
    ____________________________________________________________________________________________________
    conv1 (Conv2D)                   (None, 23, 23, 64)    1792        input_1[0][0]
    ____________________________________________________________________________________________________
    relu_conv1 (Activation)          (None, 23, 23, 64)    0           conv1[0][0]
    ____________________________________________________________________________________________________
    pool1 (MaxPooling2D)             (None, 11, 11, 64)    0           relu_conv1[0][0]
    ____________________________________________________________________________________________________
    fire2/squeeze1x1 (Conv2D)        (None, 11, 11, 16)    1040        pool1[0][0]
    ____________________________________________________________________________________________________
    fire2/relu_squeeze1x1 (Activatio (None, 11, 11, 16)    0           fire2/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire2/expand1x1 (Conv2D)         (None, 11, 11, 64)    1088        fire2/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire2/expand3x3 (Conv2D)         (None, 11, 11, 64)    9280        fire2/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire2/relu_expand1x1 (Activation (None, 11, 11, 64)    0           fire2/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire2/relu_expand3x3 (Activation (None, 11, 11, 64)    0           fire2/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire2/concat (Concatenate)       (None, 11, 11, 128)   0           fire2/relu_expand1x1[0][0]
                                                                       fire2/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire3/squeeze1x1 (Conv2D)        (None, 11, 11, 16)    2064        fire2/concat[0][0]
    ____________________________________________________________________________________________________
    fire3/relu_squeeze1x1 (Activatio (None, 11, 11, 16)    0           fire3/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire3/expand1x1 (Conv2D)         (None, 11, 11, 64)    1088        fire3/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire3/expand3x3 (Conv2D)         (None, 11, 11, 64)    9280        fire3/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire3/relu_expand1x1 (Activation (None, 11, 11, 64)    0           fire3/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire3/relu_expand3x3 (Activation (None, 11, 11, 64)    0           fire3/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire3/concat (Concatenate)       (None, 11, 11, 128)   0           fire3/relu_expand1x1[0][0]
                                                                       fire3/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    pool3 (MaxPooling2D)             (None, 5, 5, 128)     0           fire3/concat[0][0]
    ____________________________________________________________________________________________________
    fire4/squeeze1x1 (Conv2D)        (None, 5, 5, 32)      4128        pool3[0][0]
    ____________________________________________________________________________________________________
    fire4/relu_squeeze1x1 (Activatio (None, 5, 5, 32)      0           fire4/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire4/expand1x1 (Conv2D)         (None, 5, 5, 128)     4224        fire4/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire4/expand3x3 (Conv2D)         (None, 5, 5, 128)     36992       fire4/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire4/relu_expand1x1 (Activation (None, 5, 5, 128)     0           fire4/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire4/relu_expand3x3 (Activation (None, 5, 5, 128)     0           fire4/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire4/concat (Concatenate)       (None, 5, 5, 256)     0           fire4/relu_expand1x1[0][0]
                                                                       fire4/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire5/squeeze1x1 (Conv2D)        (None, 5, 5, 32)      8224        fire4/concat[0][0]
    ____________________________________________________________________________________________________
    fire5/relu_squeeze1x1 (Activatio (None, 5, 5, 32)      0           fire5/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire5/expand1x1 (Conv2D)         (None, 5, 5, 128)     4224        fire5/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire5/expand3x3 (Conv2D)         (None, 5, 5, 128)     36992       fire5/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire5/relu_expand1x1 (Activation (None, 5, 5, 128)     0           fire5/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire5/relu_expand3x3 (Activation (None, 5, 5, 128)     0           fire5/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire5/concat (Concatenate)       (None, 5, 5, 256)     0           fire5/relu_expand1x1[0][0]
                                                                       fire5/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    pool5 (MaxPooling2D)             (None, 2, 2, 256)     0           fire5/concat[0][0]
    ____________________________________________________________________________________________________
    fire6/squeeze1x1 (Conv2D)        (None, 2, 2, 48)      12336       pool5[0][0]
    ____________________________________________________________________________________________________
    fire6/relu_squeeze1x1 (Activatio (None, 2, 2, 48)      0           fire6/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire6/expand1x1 (Conv2D)         (None, 2, 2, 192)     9408        fire6/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire6/expand3x3 (Conv2D)         (None, 2, 2, 192)     83136       fire6/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire6/relu_expand1x1 (Activation (None, 2, 2, 192)     0           fire6/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire6/relu_expand3x3 (Activation (None, 2, 2, 192)     0           fire6/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire6/concat (Concatenate)       (None, 2, 2, 384)     0           fire6/relu_expand1x1[0][0]
                                                                       fire6/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire7/squeeze1x1 (Conv2D)        (None, 2, 2, 48)      18480       fire6/concat[0][0]
    ____________________________________________________________________________________________________
    fire7/relu_squeeze1x1 (Activatio (None, 2, 2, 48)      0           fire7/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire7/expand1x1 (Conv2D)         (None, 2, 2, 192)     9408        fire7/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire7/expand3x3 (Conv2D)         (None, 2, 2, 192)     83136       fire7/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire7/relu_expand1x1 (Activation (None, 2, 2, 192)     0           fire7/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire7/relu_expand3x3 (Activation (None, 2, 2, 192)     0           fire7/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire7/concat (Concatenate)       (None, 2, 2, 384)     0           fire7/relu_expand1x1[0][0]
                                                                       fire7/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire8/squeeze1x1 (Conv2D)        (None, 2, 2, 64)      24640       fire7/concat[0][0]
    ____________________________________________________________________________________________________
    fire8/relu_squeeze1x1 (Activatio (None, 2, 2, 64)      0           fire8/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire8/expand1x1 (Conv2D)         (None, 2, 2, 256)     16640       fire8/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire8/expand3x3 (Conv2D)         (None, 2, 2, 256)     147712      fire8/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire8/relu_expand1x1 (Activation (None, 2, 2, 256)     0           fire8/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire8/relu_expand3x3 (Activation (None, 2, 2, 256)     0           fire8/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire8/concat (Concatenate)       (None, 2, 2, 512)     0           fire8/relu_expand1x1[0][0]
                                                                       fire8/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire9/squeeze1x1 (Conv2D)        (None, 2, 2, 64)      32832       fire8/concat[0][0]
    ____________________________________________________________________________________________________
    fire9/relu_squeeze1x1 (Activatio (None, 2, 2, 64)      0           fire9/squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire9/expand1x1 (Conv2D)         (None, 2, 2, 256)     16640       fire9/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire9/expand3x3 (Conv2D)         (None, 2, 2, 256)     147712      fire9/relu_squeeze1x1[0][0]
    ____________________________________________________________________________________________________
    fire9/relu_expand1x1 (Activation (None, 2, 2, 256)     0           fire9/expand1x1[0][0]
    ____________________________________________________________________________________________________
    fire9/relu_expand3x3 (Activation (None, 2, 2, 256)     0           fire9/expand3x3[0][0]
    ____________________________________________________________________________________________________
    fire9/concat (Concatenate)       (None, 2, 2, 512)     0           fire9/relu_expand1x1[0][0]
                                                                       fire9/relu_expand3x3[0][0]
    ____________________________________________________________________________________________________
    drop9 (Dropout)                  (None, 2, 2, 512)     0           fire9/concat[0][0]
    ____________________________________________________________________________________________________
    conv10 (Conv2D)                  (None, 2, 2, 1000)    513000      drop9[0][0]
    ____________________________________________________________________________________________________
    relu_conv10 (Activation)         (None, 2, 2, 1000)    0           conv10[0][0]
    ____________________________________________________________________________________________________
    global_average_pooling2d_1 (Glob (None, 1000)          0           relu_conv10[0][0]
    ____________________________________________________________________________________________________
    loss (Activation)                (None, 1000)          0           global_average_pooling2d_1[0][0]
    ====================================================================================================
    Total params: 1,235,496
    Trainable params: 1,235,496
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    if input_tensor is None:
        raw_img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            raw_img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            raw_img_input = input_tensor

    if use_bn_on_input:
        img_input = BatchNormalization()(raw_img_input)
    else:
        img_input = raw_img_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride),
                      padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation(last_activation, name='loss')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = raw_img_input

    model = Model(inputs, out, name='squeezenet')

    # load weights
    if weights == 'imagenet':

        weights_path = get_file(
            'squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models')
        model.load_weights(weights_path, by_name = load_by_name)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model
