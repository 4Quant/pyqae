import inspect
import json
import marshal
import os
import types
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

STD_IMPORT_CODE = """from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np
MAKE_RGB_LAYER = False
USE_B_CONV = False
"""


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    A class to serialize numpy arrays (very inefficiently) as JSON
    It turns them into nested lists which can easily be read
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


get_temp_dir = lambda: TemporaryDirectory()


def export_keras_as_tf(in_model,  # type: keras.models.Model
                       model_out_dir,  # type: str
                       prefix,  # type: str
                       export_version,  # type: int
                       make_ascii=False,
                       lock_learning_phase=True,
                       image_labels=None
                       # type: Optional[List[Tuple[List[str]]]]
                       ):
    # type: (...) -> str
    """
    Export a Keras model as a static tensorflow protobuf
    :param in_model:  the model to export
    :param model_out_dir:  the output directory
    :param prefix:  the name of the model
    :param export_version: version number of the model (will fail if already exists)
    :param make_ascii: make an ascii version of the graph
    :return: the name of the folder where the graph is saved
    here is an example showing the IO working on a non-standard Keras model
    >>> from pyqae.dnn.features import PhiComGrid3DLayer, CCLTimeZoomLayer
    >>> from keras.models import Sequential
    >>> t_model = Sequential()
    >>> t_model.add(PhiComGrid3DLayer(z_rad=0.0, include_r = True, include_ir = True, input_shape=(None, None, None, 1), name='PhiGrid'))
    >>> model_dir = get_temp_dir()
    >>> m_dir = export_keras_as_tf(t_model, model_dir.name, 'test', 0, False)
    output nodes names are:  ['PhiGrid/add_com_phi_grid_3d/phi_coord_3d/concat']
    Converted 0 variables to const ops.
    saved the constant graph (ready for inference) at:  constant_graph_weights.pb
    >>> t_img = np.ones((1, 3, 3, 3, 1))
    >>> o_iter = list(evaluate_tf_model(m_dir, [t_img], show_nodes = True))
    Ops: 287
    Tensor("PhiGrid_input:0", shape=(?, ?, ?, ?, 1), dtype=float32, device=/device:CPU:0)
    Tensor("PhiGrid/add_com_phi_grid_3d/phi_coord_3d/concat:0", shape=(?, ?, ?, ?, 5), dtype=float32, device=/device:CPU:0)
    >>> for (a,_,c) in o_iter[0]: print(a, c.shape)
    PhiGrid/add_com_phi_grid_3d/phi_coord_3d/concat:0 (1, 3, 3, 3, 5)
    >>> n_model = Sequential()
    >>> n_model.add(CCLTimeZoomLayer(0, 2, 4, 5, input_shape=(3, 3, 1), name='CC'))
    >>> n_dir = export_keras_as_tf(n_model, model_dir.name, 'test', 1, False)
    output nodes names are:  ['CC/label_zoom_time/scipy_batch_label_zoom']
    Converted 0 variables to const ops.
    saved the constant graph (ready for inference) at:  constant_graph_weights.pb
    >>> t_img = np.ones((1, 3, 3, 1))
    >>> o_iter = list(evaluate_tf_model(n_dir, [t_img], show_nodes = True))
    Ops: 2
    Tensor("CC_input:0", shape=(?, 3, 3, 1), dtype=float32, device=/device:CPU:0)
    Tensor("CC/label_zoom_time/scipy_batch_label_zoom:0", dtype=float32, device=/device:CPU:0)
    >>> for (a,_,c) in o_iter[0]: print(a, c.shape)
    CC/label_zoom_time/scipy_batch_label_zoom:0 (1, 2, 4, 5, 1)
    >>> model_dir.cleanup()
    """
    from keras import backend as K
    assert K.backend() == "tensorflow", \
        "Export as TF only works in TF mode {}".format(K.backend())
    export_path = os.path.join(model_out_dir,
                               '%s_model_%04d' % (prefix, export_version))
    os.makedirs(export_path, exist_ok=False)

    if lock_learning_phase:
        K.set_learning_phase(0)  # TODO: this doesnt seem to work yet so we
        # TODO:manually hack it in the next one by setting all of these in the feed dict

    num_inputs = len(in_model.inputs)
    num_outputs = len(in_model.outputs)
    # NOTE: these need be op's not tensors
    pred_node_names = [in_model.get_output_at(i).op.name for i in range(
        num_outputs)]

    print('output nodes names are: ', pred_node_names)
    sess = K.get_session()

    if make_ascii:
        f = 'only_the_graph_def.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), export_path, f,
                             as_text=True)
        print('saved the graph definition in ascii format at: ',
              f)
    output_graph_name = 'constant_graph_weights.pb'
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               pred_node_names)
    graph_io.write_graph(constant_graph, export_path, output_graph_name,
                         as_text=False)
    print('saved the constant graph (ready for inference) at: ',
          output_graph_name)
    extra_keys = dict()
    if image_labels is not None:
        extra_keys['image_labels'] = image_labels
    with open(os.path.join(export_path, 'graph.json'), 'w') as w:
        in_shapes = in_model.input_shape
        if type(in_shapes) is tuple:
            in_shapes = [in_shapes]
        out_shapes = in_model.output_shape
        if type(out_shapes) is tuple:
            out_shapes = [out_shapes]
        json.dump(dict(input=[j.name for j in in_model.inputs],
                       output=[j.name for j in in_model.outputs],
                       in_shapes=in_shapes,
                       out_shapes=out_shapes,
                       graph_def=output_graph_name,
                       **extra_keys),
                  w)
    return export_path


from itertools import chain
from tensorflow.python.platform import gfile
from tqdm import tqdm


def evaluate_tf_model(in_model_dir,  # type: str
                      in_inputs,  # type: List[np.ndarray]
                      show_nodes=False,
                      batch_size=None,
                      graph_file_name='graph.json',
                      tf_device='/cpu:0'):
    # type: (...) -> Generator[List[Tuple[str, List[str], np.ndarray]]]
    """
    Evaluate from a saved tensorflow model
    :param in_model_dir: the name of the json file containing the information
    :param in_inputs: the inputs to the model
    :param show_nodes: show the relevant input and output nodes
    :param batch_size: run the job in batches (if necessary)
    :param tf_device: the device to use for importing the graph
    :return: a generator containing the results per batch
    """
    with open(os.path.join(in_model_dir, graph_file_name), 'r') as r:
        model_desc = json.load(r)

    def _ch_gen():
        i = 0
        while True:
            yield 'Ch_{}'.format(i)
            i += 1

    fake_labels = [[out_layer, _ch_gen()] for out_layer in
                   model_desc['output']]
    out_labels = [(out_name, out_layers) for out_name, out_layers in
                  model_desc.get('image_labels', fake_labels)]

    with gfile.FastGFile(os.path.join(in_model_dir,
                                      model_desc['graph_def']),
                         'rb') as f:
        with tf.Graph().as_default() as g:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with tf.device(tf_device):
                tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=g) as sess:
        if show_nodes:
            print('Ops:', len(sess.graph.get_operations()))
            for input_node in model_desc['input'] + model_desc['output']:
                input_x = sess.graph.get_tensor_by_name(input_node)
                print(input_x)
        inputs = [sess.graph.get_tensor_by_name(c_node) for c_node in
                  model_desc['input']]
        outputs = [sess.graph.get_tensor_by_name(c_node) for c_node in
                   model_desc['output']]
        feed_dict = {}
        # this code is for improperly saved keras models so the training
        # phase is set otherwise it won't evaluate correctly (harmless for
        # everything else)
        klp = 'keras_learning_phase'
        kl_phases = list(chain(
            *[c_op.values() for c_op in sess.graph.get_operations() if
              klp in c_op.name]))
        for c_phase in kl_phases:
            feed_dict[c_phase] = False

        if batch_size is None:
            for c_in, c_img in zip(inputs, in_inputs):
                feed_dict[c_in] = c_img
                c_output = sess.run(outputs, feed_dict=feed_dict)
                yield [(c_label, ch_labels, out_img)
                       for (c_label, ch_labels), out_img
                       in zip(out_labels, c_output)]
        else:
            n_items = in_inputs[0].shape[0]
            for i in tqdm(range(0, n_items, batch_size)):
                for c_in, c_img in zip(inputs, in_inputs):
                    feed_dict[c_in] = c_img[i:(i + batch_size)]
                    c_output = sess.run(outputs, feed_dict=feed_dict)
                    yield [(c_label, ch_labels, out_img)
                           for (c_label, ch_labels), out_img
                           in zip(out_labels, c_output)]


class ImageModel(object):
    """
    A class for making image-based predictions easier
    """

    def __init__(self, imodel, preproc_fun=lambda x: x):
        self.__rmodel = imodel
        self.input_size = imodel.get_input_shape_at(0)
        self.preproc_fun = preproc_fun

    def predict(self, in_image, prepend_shape=True, pad_shape=False, *args,
                **kwargs):
        """
        Predict the output for an input images

        Parameters
        ----------
        in_image : numpy array 2D, 3D, or 4D
            containing image to be processed

        prepend_shape : boolean, optional, default True
            Add dimensions to beginning of shape of size 1 if they are missing
        
        pad_shape : boolean, optional, default False
            Zero-pad dimensions so the image matches the model size
        """
        pp_image = self.preproc_fun(in_image)
        if prepend_shape:
            while len(pp_image.shape) < len(self.input_size):
                pp_image = np.expand_dims(in_image, 0)
        assert len(pp_image.shape) == len(
            self.input_size), "Image dimensions do not match model: {} != {}".format(
            pp_image.shape, self.input_size)

        for i, (cur_size, mod_size) in enumerate(
                zip(pp_image.shape, self.input_size)):
            if mod_size is not None:  # batch dimension is none
                if pad_shape:
                    if cur_size < mod_size:
                        cur_size = mod_size  ##TODO actually pad it with np.pad
                assert cur_size == mod_size, "Image at dimension at {} must match ({}, {})".format(
                    i, pp_image.shape,
                    self.input_size)

        return self.__rmodel.predict(pp_image, *args, **kwargs)


def write_msh_model(file_path, in_model, model_fcns, model_fcn_name=None,
                    in_preproc_fun=lambda x: x, make_json=True):
    """
    Save everything needed to create a model in a portable file
    
    Parameters
    ----------
    file_path : str
        The path to save the marshal file

    in_model : model (must have .layers)
        the model to be serialized
        
    model_fcns : list [functions]
        The list of functions to be serialized required for making the model
    
    model_fcn_name : str
        The name of the function which can be called with the size to create a network
    
    in_preproc_fun : function
        A self-contained function for preprocessing the input so the network can be used properly (rescaling / normalization) 
    
    make_json: bool
        Make a json copy (less version specific than the marshalling but very bloated)
    """
    fcn_names = [cfunc.func_name for cfunc in model_fcns]
    if model_fcn_name is not None:
        assert model_fcn_name in fcn_names, "Model function name must be in offical model name list ({}) not in {}".format(
            model_fcn_name, fcn_names)
    else:
        model_fcn_name = fcn_names[-1]
        print('Assuming model creation function is', model_fcn_name)
    with open('{}.msh'.format(file_path), 'wb') as wf:
        imp_code = STD_IMPORT_CODE
        marshal.dump(imp_code, wf)
        marshal.dump(model_fcn_name, wf)
        marshal.dump(len(model_fcns), wf)
        clines = []
        for cfunc in model_fcns:
            marshal.dump(cfunc.func_name, wf)
            marshal.dump(cfunc.func_code, wf)
        marshal.dump(in_preproc_fun.func_code, wf)
        weight_dict = {}
        for i, ilayer in enumerate(in_model.layers):
            weight_dict[ilayer.name] = [i, [cw.tolist() for cw in
                                            ilayer.get_weights()]]
        marshal.dump(weight_dict, wf)
    if make_json:
        out_dict = {'imp_code': STD_IMPORT_CODE,
                    'model_fcn_name': model_fcn_name,
                    'model_fcns': [
                        (cfunc.func_name, inspect.getsourcelines(cfunc)) for
                        cfunc in model_fcns],
                    'in_preproc_fun': inspect.getsourcelines(in_preproc_fun),
                    'weights': weight_dict,
                    'config': in_model.get_config()}
        with open('{}.json'.format(file_path), 'w') as wf:
            json.dump(out_dict, wf, cls=NumpyAwareJSONEncoder)


def build_msh_model(file_path, in_shape=(1, 3, 128, 128),
                    allow_mismatch=False):
    """
    Loads a model and weights from a given path
    
    Parameters
    ----------
    file_path : str
        The path of the network to load
    
    in_shape : tuple
        The dimensions of the input for the network
    
    allow_mismatch : bool, optionional, default False
        Whether or not to allow mismatched layers sizes (throws an assertion error)
    """
    g_var, l_var = {}, {}
    with open('{}.msh'.format(file_path), 'rb') as rf:
        imp_code = marshal.load(rf)
        exec(imp_code, g_var, l_var)
        mdl_spawn_fcn_name = marshal.load(rf)
        assert type(
            mdl_spawn_fcn_name) is str, "Model name must be string -> {}".format(
            mdl_spawn_fcn_name)
        func_count = marshal.load(rf)
        func_dict = {}
        for i in range(func_count):
            try:
                f_name = marshal.load(rf)
                code = marshal.load(rf)
                g_var[f_name] = types.FunctionType(code, g_var, f_name)

            except:
                print("Error reading function", i)
                break

        preproc_fun = types.FunctionType(marshal.load(rf), g_var,
                                         "preproc_fun")
        out_weights = marshal.load(rf)
        assert type(
            out_weights) is dict, "The weights must be stored in a dictionary"
    for f_name, f_func in func_dict.items():
        # exec('{} = f_func'.format(f_name))
        print(f_func)
    g_var['np'] = np
    for i, v in l_var.items(): g_var[i] = v
    exec('c_model = {}({})'.format(mdl_spawn_fcn_name, in_shape), g_var)
    c_model = g_var['c_model']
    for (lay_name, (lay_idx, lay_weights)) in out_weights.items():
        assert lay_name == c_model.layers[
            lay_idx].name, "Names do not match {} != {}".format(lay_name,
                                                                c_model.layers[
                                                                    lay_idx].name)
        c_weights = c_model.layers[lay_idx].get_weights()
        assert len(c_weights) == len(
            lay_weights), "{} layer weights must have same number".format(
            lay_name)
        c_shapes = [cw.shape for cw in c_weights]
        lay_weights = map(np.array,
                          lay_weights)  # make sure all of the arrays are numpy not python
        l_shapes = [lw.shape for lw in lay_weights]
        match_mat = all([cs == ls for cs, ls in zip(c_shapes, l_shapes)])
        mis_msg = "{} layer shapes must match {} != {}".format(lay_name,
                                                               c_shapes,
                                                               l_shapes)
        if not allow_mismatch:
            assert match_mat, mis_msg
        if match_mat:
            c_model.layers[lay_idx].set_weights(lay_weights)
        else:
            print(mis_msg)
    return ImageModel(c_model, preproc_fun)


class NetEncoder(object):
    """Encoder class.
    Weights are serialized sequentially from the Keras flattened_layers representation
    into:
        - `weights`: a binary string representing the raw data bytes in float32
            of all weights, sequentially concatenated.
        - `metadata`: a list containing the byte length and tensor shape,
            so that the original tensors can be reconstructed
    """

    def __init__(self, out_model):
        self.out_model = out_model
        self.weights = b''
        self.metadata = []

    def serialize(self):
        """serialize method.
        Strategy for extracting the weights is adapted from the
        load_weights_from_hdf5_group method of the Container class:
        see https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L2505-L2585
        """
        with NamedTemporaryFile(suffix='.h5') as c_weight_file:
            w_name = c_weight_file.name
            self.out_model.save_weights(w_name)
            hdf5_file = h5py.File(w_name, mode='r')
            if 'layer_names' not in hdf5_file.attrs and 'model_weights' in hdf5_file:
                f = hdf5_file['model_weights']
            else:
                f = hdf5_file

            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
            offset = 0
            for layer_name in layer_names:
                g = f[layer_name]
                weight_names = [n.decode('utf8') for n in
                                g.attrs['weight_names']]
                if len(weight_names):
                    # noinspection PyDictCreation
                    for weight_name in weight_names:
                        meta = {}
                        meta['layer_name'] = layer_name
                        meta['weight_name'] = weight_name
                        weight_value = g[weight_name].value
                        bytearr = weight_value.astype(np.float32).tobytes()
                        self.weights += bytearr
                        meta['offset'] = offset
                        meta['length'] = len(bytearr) // 4
                        meta['shape'] = list(weight_value.shape)
                        meta['type'] = 'float32'
                        self.metadata.append(meta)
                        offset += len(bytearr)

            hdf5_file.close()

    def save(self, out_dir, out_prefix='nn'):
        """Saves weights data (binary) and weights metadata (json)
        """
        model_filepath = '{}_model.json'.format(
            os.path.join(out_dir, out_prefix))
        with open(model_filepath, 'w') as f:
            f.write(self.out_model.to_json())
        weights_filepath = '{}_weights.buf'.format(
            os.path.join(out_dir, out_prefix))
        with open(weights_filepath, mode='wb') as f:
            f.write(self.weights)
        metadata_filepath = '{}_metadata.json'.format(
            os.path.join(out_dir, out_prefix))
        with open(metadata_filepath, mode='w') as f:
            json.dump(self.metadata, f)
        return {'model_path': model_filepath, 'weights_path': weights_filepath,
                'metadata_path': metadata_filepath}

    @staticmethod
    def saveModel(model, out_dir, out_prefix='nn'):
        """Saves everything in a single function
        """
        new_net = NetEncoder(model)
        new_net.serialize()
        return new_net.save(out_dir, out_prefix)


import pandas as pd


def _export_edges(in_model):
    # type: (keras.models.Model) -> pd.DataFrame
    """
    Convert a Keras model into the list of edges between nodes
    :param in_model: 
    :return: 
    >>> from keras.applications import VGG16
    >>> vnet = VGG16(True, weights = None)
    >>> edge_df = _export_edges(vnet)
    >>> edge_df.shape
    (22, 2)
    >>> edge_df.head(2)
               from            to
    0       input_1  block1_conv1
    1  block1_conv1  block1_conv2
    """

    def get_var_count(c_layer):
        return c_layer['config'].get('filters', 0) * np.prod(
            c_layer['config'].get('kernel_size', (0, 0, 0)))

    def _unwrap(t_model):
        for ilayer in t_model.get_config()['layers']:
            for inode in ilayer['inbound_nodes']:
                for jnode in inode:
                    yield {'to': ilayer['name'],
                           'from': jnode[0]
                           }

    return pd.DataFrame(_unwrap(in_model))


def _export_nodes(t_model, input_shape=None):
    # type: (keras.models.Model, Optional[List[int]]) -> pd.DataFrame
    """
    Export all of the nodes from a Keras neural networ
    :param t_model: 
    :param input_shape: 
    :return: DataFrame with all nodes
    >>> from keras.applications import VGG16
    >>> vnet = VGG16(True, weights = None)
    >>> node_df = _export_nodes(vnet)
    >>> node_df.shape
    (23, 9)
    >>> node_df[['class_name', 'shape_x', 'shape_c', 'param_count']].head(2)
       class_name  shape_x  shape_c  param_count
    0  InputLayer    224.0        3            0
    1      Conv2D    224.0       64         1792
    """
    all_nodes = {i.name: i for i in t_model.layers}

    def get_shape(ilay):
        if input_shape is not None:
            try:
                out_shape = ilay.compute_output_shape(input_shape=input_shape)
            except:
                out_shape = ilay.output_shape
        else:
            out_shape = ilay.output_shape
        if len(out_shape) == 1:
            shape_str = 'n'
        elif len(out_shape) == 2:
            shape_str = 'nc'
        elif len(out_shape) == 3:
            shape_str = 'nxc'
        elif len(out_shape) == 4:
            shape_str = 'nxyc'
        elif len(out_shape) == 5:
            shape_str = 'nxyzc'
        else:
            shape_str = ['%02d' % i for i in range(len(out_shape))]
        return {'shape_{}'.format(k_name): k for k_name, k in
                zip(shape_str, out_shape)}

    return pd.DataFrame([dict(name=i.name,
                              channels=i.output_shape[-1],
                              dof_count=sum(
                                  [np.prod(cw.get_shape().as_list()) for cw in
                                   i.trainable_weights]),
                              class_name=type(i).__name__,
                              param_count=i.count_params(),
                              **get_shape(i)
                              ) for i in t_model.layers])


def export_network_csv(in_model, out_base_name, input_shape=None):
    """
    Export a network as two csv files
    :param in_model: 
    :param out_base_name: 
    :param input_shape: 
    :return: 
    """
    out_df = _export_edges(in_model)
    out_df.to_csv('{}_edge.csv'.format(out_base_name), index=False)
    node_df = _export_nodes(in_model, input_shape)
    node_df.to_csv('{}_node.csv'.format(out_base_name), index=False)
