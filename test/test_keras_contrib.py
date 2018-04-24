"""
Since the keras contribution package is less robustly tested and
compatiblity with Keras isn't always kept up to date, we have our own set of tests to ensure various layers that we need work
"""

import pytest
import os
pytestmark = pytest.mark.usefixtures("eng")
_res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'resources')
def test_pelu_layer(eng):
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy as np
    from keras_contrib.layers.advanced_activations import PELU

    # Create the Keras model, including the PELU advanced activation
    model = Sequential()
    model.add(Dense(100, input_shape=(10,)))
    model.add(PELU())

    # Compile and fit on random data
    model.compile(loss='mse', optimizer='adam')
    model.fit(x=np.random.random((100, 10)), y=np.random.random((100, 100)),
              epochs=5, verbose=0)

def test_instancenorm(eng):
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy as np
    from keras_contrib.layers.normalization import InstanceNormalization

    # Create the Keras model, and InstanceNormalization
    model = Sequential()
    model.add(InstanceNormalization(input_shape=(10,)))
    model.add(Dense(100))

    # Compile and fit on random data
    model.compile(loss='mse', optimizer='adam')
    model.fit(x=np.random.random((100, 10)),
              y=np.random.random((100, 100)),
              epochs=5,
              verbose=0)
