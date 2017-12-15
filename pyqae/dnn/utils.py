from keras import backend as K
import numpy as np

def predict_with_dropout(c_model, c_inputs, iterations=5):
    """
    A way for executing models with dropout and averaging the results to
    obtain a better estimate of confidence and sensitivity for a given model / prediction
    :param c_model:
    :param c_inputs:
    :param iterations:
    :return:

    Example
    =======
    >>> np.random.seed(2017)
    >>> from keras.models import Sequential
    >>> from keras.layers import GaussianNoise, Dropout, SpatialDropout2D
    >>> from pyqae.utils import pprint
    >>> simple_model = Sequential()
    >>> simple_model.add(GaussianNoise(0.1, input_shape = (1,)))
    >>> pprint(simple_model.predict(np.ones((2,1))))
    [[ 1.]
     [ 1.]]
    >>> out = predict_with_dropout(simple_model, [np.ones((2,1))])
    >>> len(out), np.std(out[0])>1e-3
    (1, True)
    >>> dropout_model = Sequential()
    >>> dropout_model.add(Dropout(0.5, input_shape = (10,)))
    >>> dout = predict_with_dropout(dropout_model, [np.ones((1,10))], 1)
    >>> pprint(dout[0])
    [[ 2.  0.  2.  0.  2.  0.  0.  0.  0.  2.]]
    >>> dropout2d_model = Sequential()
    >>> dropout2d_model.add(SpatialDropout2D(0.5, input_shape = (4, 4, 3)))
    >>> d2_in = np.arange(48).reshape((1, 4, 4, 3))
    >>> d2out = predict_with_dropout(dropout2d_model, [d2_in], 1)
    >>> pprint(d2out[0][0,:,:,0])
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    >>> pprint(d2out[0][0,:,:,1])
    [[  2.   8.  14.  20.]
     [ 26.  32.  38.  44.]
     [ 50.  56.  62.  68.]
     [ 74.  80.  86.  92.]]
    >>> pprint(d2out[0][0,:,:,2])
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    >>> pprint(np.sum(d2out[0], (0, 1, 2)))
    [   0.  752.    0.]
    >>> np.abs(d2out[0].mean() - d2_in.mean())>0.1
    True
    """
    raw_model_fcn = K.function(c_model.inputs + [K.learning_phase()],
                               c_model.outputs)
    out_vals = None
    for i in range(iterations):
        c_vals = raw_model_fcn(c_inputs + [1])
        # add to output as list of variables with a list of observations inside out_vals[var_num][observation_number]
        if out_vals is None:
            out_vals = [[x] for x in c_vals]
        else:
            for j, c_val in enumerate(c_vals):
                out_vals[j] += [c_val]

    return [np.sum(np.stack(x, 0), 0) / iterations for x in out_vals]
