"""
A collection of simple standard layers which aren't yet part of the brain or keras-contrib branches

"""
from keras.engine import Layer, InputSpec

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K


class Scale(Layer):
    '''Custom Layer for DenseNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases learned.
    code from: https://raw.githubusercontent.com/flyyufelix/DenseNet-Keras/master/custom_layers.py

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    >>> from keras.models import Sequential
    >>> from keras.layers import RepeatVector
    >>> from pyqae.utils import pprint
    >>> import numpy as np
    >>> my_model = Sequential()
    >>> my_model.add(RepeatVector(2, input_shape = (4,)))
    >>> pprint(my_model.predict(np.ones((1,4))))
    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]
    >>> w = [0.5*np.ones((4,)), 0.1*np.zeros((4,))]
    >>> my_model.add(Scale(weights = w))
    >>> pprint(my_model.predict(np.ones((1,4))))
    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]
    >>> my_model.layers[-1].get_config()
    {'axis': -1, 'momentum': 0.9, 'trainable': True, 'name': 'scale_1'}
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero',
                 gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape),
                                name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape),
                               name='{}_beta'.format(self.name))
        # self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        # self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta,
                                                                     broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
