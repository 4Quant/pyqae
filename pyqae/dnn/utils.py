import numpy as np
from keras import backend as K


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


def _virt_in_layer(name, batch_input_shape, inbound_nodes=None):
    if inbound_nodes is None:
        inbound_nodes = []
    return {'class_name': 'InputLayer',
            'config': {'batch_input_shape': batch_input_shape,
                       'dtype': 'float32',
                       'name': name,
                       'sparse': False},
            'inbound_nodes': inbound_nodes,
            'name': name}


def seq_to_func_dict(in_seq_dict):
    """
    Convert the dictionary representations of a sequential model into a
    functional or graph-based model

    :param in_seq_dict:
    :return:
    """
    assert in_seq_dict[
               'class_name'] == 'Sequential', 'Requires sequential model, not {}'.format(
        in_seq_dict['class_name'])
    new_func_model = in_seq_dict.copy()
    new_func_model['class_name'] = 'Model'
    old_layers = in_seq_dict['config']
    # take the shape from the first layer
    old_in_shape = old_layers[0]['config']['batch_input_shape']
    in_name = 'in_%08d' % int(np.random.uniform(0, 1e6))
    new_layers = [_virt_in_layer(in_name, batch_input_shape=old_in_shape)]
    last_name = in_name
    for n_layer in old_layers:
        if 'name' not in n_layer.keys():
            if isinstance(n_layer['config'], dict):
                n_layer['name'] = n_layer['config']['name']
            else:
                n_layer['name'] = 'layer_%08d' % int(np.random.uniform(0, 1e6))

        n_layer['inbound_nodes'] = [[[last_name, 0, 0, {}]]]
        last_name = n_layer['name']
        new_layers += [n_layer]
    # layer idx are an oversimplification that only works for top-level models
    new_func_model['config'] = dict(layers=new_layers,
                                    input_layers=[[in_name, 0, 0]],
                                    output_layers=[[last_name, 0, 0]])

    return new_func_model


import json


def seq_to_func_model(in_seq_model, copy_weights=True):
    # type: (keras.models.Sequential, Optional[bool]) -> keras.models.Model
    """
    Convert a sequential version of a model to a graph-based (useful for
    serialization
    :param in_seq_model:
    :return:
    >>> from keras import models, layers
    >>> test_model = models.Sequential()
    >>> test_model.add(layers.Lambda(lambda x: 0*x, input_shape = (2, 3)))
    >>> np.random.seed(2018)
    >>> g_model = seq_to_func_model(test_model)
    >>> g_model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    in_00882349 (InputLayer)     (None, 2, 3)              0
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 2, 3)              0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    >>> g_model.predict((np.ones((1, 2, 3)))).ravel()
    array([ 0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)
    >>> d_model = models.Sequential()
    >>> d_model.add(layers.BatchNormalization(input_shape = (5, 4, 3)))
    >>> d_s_model = models.Sequential()
    >>> d_s_model.add(layers.BatchNormalization(name = 'junk', input_shape = (5,  4, 3)))
    >>> d_model.add(d_s_model)
    >>> from pyqae.utils import get_error
    >>> get_error(seq_to_func_model, in_seq_model = d_model) # nested models
    'Graph disconnected: cannot obtain value for tensor Tensor("junk_input_1:0", shape=(?, 5, 4, 3), dtype=float32) at layer "junk_input". The following previous layers were accessed without issue: []'
    """
    from keras.models import model_from_json
    py_seq_model = json.loads(in_seq_model.to_json())
    py_graph_model = seq_to_func_dict(py_seq_model)
    graph_model = model_from_json(json.dumps(py_graph_model))
    if copy_weights:
        for i_layer, o_layer in zip(in_seq_model.layers,
                                    graph_model.layers[1:]):
            o_layer.set_weights(i_layer.get_weights())
    return graph_model
