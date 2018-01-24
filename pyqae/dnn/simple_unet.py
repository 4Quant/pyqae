from keras.layers import Input, Conv2D, BatchNormalization, \
    Activation, Deconv2D, MaxPool2D, concatenate
from keras.models import Model


def make_model(in_shape, layers, initial_depth, prefix=''):
    # type: (Tuple[int, int, int], int, int, str) -> keras.models.Model
    """
    A simple, crop-free UNET for quick experimentation and baseline references
    :param in_shape:
    :param layers:
    :param initial_depth:
    :param prefix:
    :return:
    >>> simple_model = make_model((32, 32, 1), 2, 8, prefix='HI')
    >>> len(simple_model.layers)
    20
    >>> simple_model.summary()
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    UNET_HI_Input (InputLayer)      (None, 32, 32, 1)    0
    __________________________________________________________________________________________________
    CONV_HI_0 (Conv2D)              (None, 32, 32, 8)    80          UNET_HI_Input[0][0]
    __________________________________________________________________________________________________
    BN_HI_0 (BatchNormalization)    (None, 32, 32, 8)    32          CONV_HI_0[0][0]
    __________________________________________________________________________________________________
    RELU_HI_0 (Activation)          (None, 32, 32, 8)    0           BN_HI_0[0][0]
    __________________________________________________________________________________________________
    MP_HI_0 (MaxPooling2D)          (None, 16, 16, 8)    0           RELU_HI_0[0][0]
    __________________________________________________________________________________________________
    CONV_HI_1 (Conv2D)              (None, 16, 16, 16)   1168        MP_HI_0[0][0]
    __________________________________________________________________________________________________
    BN_HI_1 (BatchNormalization)    (None, 16, 16, 16)   64          CONV_HI_1[0][0]
    __________________________________________________________________________________________________
    RELU_HI_1 (Activation)          (None, 16, 16, 16)   0           BN_HI_1[0][0]
    __________________________________________________________________________________________________
    MP_HI_1 (MaxPooling2D)          (None, 8, 8, 16)     0           RELU_HI_1[0][0]
    __________________________________________________________________________________________________
    DECONV_HI_1 (Conv2DTranspose)   (None, 16, 16, 16)   2320        MP_HI_1[0][0]
    __________________________________________________________________________________________________
    SKIP_HI_1 (Concatenate)         (None, 16, 16, 24)   0           DECONV_HI_1[0][0]
                                                                     MP_HI_0[0][0]
    __________________________________________________________________________________________________
    BN_HI_U1 (BatchNormalization)   (None, 16, 16, 24)   96          SKIP_HI_1[0][0]
    __________________________________________________________________________________________________
    RELU_HI_U1 (Activation)         (None, 16, 16, 24)   0           BN_HI_U1[0][0]
    __________________________________________________________________________________________________
    DECONV_HI_0 (Conv2DTranspose)   (None, 32, 32, 8)    1736        RELU_HI_U1[0][0]
    __________________________________________________________________________________________________
    SKIP_HI_0 (Concatenate)         (None, 32, 32, 9)    0           DECONV_HI_0[0][0]
                                                                     UNET_HI_Input[0][0]
    __________________________________________________________________________________________________
    BN_HI_U0 (BatchNormalization)   (None, 32, 32, 9)    36          SKIP_HI_0[0][0]
    __________________________________________________________________________________________________
    RELU_HI_U0 (Activation)         (None, 32, 32, 9)    0           BN_HI_U0[0][0]
    ==================================================================================================
    Total params: 5,532
    Trainable params: 5,418
    Non-trainable params: 114
    __________________________________________________________________________________________________
    """
    in_layer = Input(in_shape, name='UNET_{}_Input'.format(prefix))

    start_x = in_layer
    skip_layers = []
    for c_layer in range(layers + 1):
        skip_layers += [start_x]

        x = Conv2D(filters=initial_depth * 2 ** c_layer,
                   kernel_size=(3, 3),
                   activation='linear',
                   padding='same',
                   name='CONV_{}_{}'.format(prefix, c_layer))(start_x)
        x = BatchNormalization(name='BN_{}_{}'.format(prefix, c_layer))(x)
        x = Activation('relu', name='RELU_{}_{}'.format(prefix, c_layer))(x)
        start_x = MaxPool2D((2, 2), name='MP_{}_{}'.format(prefix, c_layer))(x)
    start_x = x
    for c_layer, c_skip in reversed(list(zip(range(layers), skip_layers))):
        x = Deconv2D(filters=initial_depth * 2 ** c_layer,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='linear',
                     padding='same',
                     name='DECONV_{}_{}'.format(prefix, c_layer)
                     )(start_x)
        x = concatenate([x, c_skip], name='SKIP_{}_{}'.format(prefix, c_layer))
        x = BatchNormalization(name='BN_{}_U{}'.format(prefix, c_layer))(x)
        x = Activation('relu', name='RELU_{}_U{}'.format(prefix, c_layer))(x)
        start_x = x
    return Model(inputs=[in_layer], outputs=[x], name='UNET_{}'.format(prefix))