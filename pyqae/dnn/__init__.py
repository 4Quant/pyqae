"""Modules related to Deep Neural Networks
"""

import keras
KERAS_2 = keras.__version__[0] == '2'
try:
    # keras 2 imports
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.merge import Concatenate
except ImportError:
    print("Keras 2 layers could not be imported defaulting to keras1")
    KERAS_2 = False

fix_name_tf = lambda name: name.replace(' ', '-').replace('[', '').replace(']', '')
