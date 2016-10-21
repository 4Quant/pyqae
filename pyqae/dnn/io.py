import marshal, types, json, inspect
import numpy as np

STD_IMPORT_CODE =  """from keras import backend as K
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
        if isinstance(obj, np.ndarray): # and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ImageModel(object):
    """
    A class for making image-based predictions easier
    """
    def __init__(self, imodel, preproc_fun = lambda x: x):
        self.__rmodel = imodel
        self.input_size = imodel.get_input_shape_at(0)
        self.preproc_fun = preproc_fun
    def predict(self, in_image, prepend_shape = True, pad_shape = False, *args, **kwargs):
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
            while len(pp_image.shape)<len(self.input_size):
                pp_image = np.expand_dims(in_image,0)
        assert len(pp_image.shape) == len(self.input_size), "Image dimensions do not match model: {} != {}".format(pp_image.shape, self.input_size)
        
        for i, (cur_size,mod_size) in enumerate(zip(pp_image.shape, self.input_size)):
            if mod_size is not None: # batch dimension is none
                if pad_shape:
                    if cur_size < mod_size:
                        cur_size = mod_size ##TODO actually pad it with np.pad
                assert cur_size == mod_size, "Image at dimension at {} must match ({}, {})".format(i,pp_image.shape, self.input_size)
        
        return self.__rmodel.predict(pp_image,*args, **kwargs)


def write_msh_model(file_path, in_model, model_fcns, model_fcn_name = None, in_preproc_fun = lambda x: x, make_json = True):
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
        assert model_fcn_name in fcn_names, "Model function name must be in offical model name list ({}) not in {}".format(model_fcn_name, fcn_names)
    else:
        model_fcn_name = fcn_names[-1]
        print('Assuming model creation function is', model_fcn_name)
    with open('{}.msh'.format(file_path),'wb') as wf:
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
            weight_dict[ilayer.name] = [i,[cw.tolist() for cw in ilayer.get_weights()]]
        marshal.dump(weight_dict,wf)
    if make_json:
        out_dict = {}
        out_dict['imp_code'] = STD_IMPORT_CODE
        out_dict['model_fcn_name'] = model_fcn_name
        out_dict['model_fcns'] = [(cfunc.func_name, inspect.getsourcelines(cfunc)) for cfunc in model_fcns]
        out_dict['in_preproc_fun'] = inspect.getsourcelines(in_preproc_fun)
        out_dict['weights'] = weight_dict
        out_dict['config'] = in_model.get_config()
        with open('{}.json'.format(file_path),'w') as wf:
            json.dump(out_dict, wf, cls = NumpyAwareJSONEncoder)


def build_msh_model(file_path, in_shape = (1,3,128,128), allow_mismatch = False):
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
    with open('{}.msh'.format(file_path),'rb') as rf:
        imp_code = marshal.load(rf)
        exec(imp_code, g_var, l_var)
        mdl_spawn_fcn_name = marshal.load(rf)
        assert type(mdl_spawn_fcn_name) is str, "Model name must be string -> {}".format(mdl_spawn_fcn_name)
        func_count = marshal.load(rf)
        func_dict = {}
        for i in range(func_count):
            try:
                f_name = marshal.load(rf)
                code = marshal.load(rf) 
                g_var[f_name] = types.FunctionType(code, g_var, f_name)
                
            except:
                print("Error reading function",i)
                break
        
        preproc_fun = types.FunctionType(marshal.load(rf), g_var, "preproc_fun")
        out_weights = marshal.load(rf)
        assert type(out_weights) is dict, "The weights must be stored in a dictionary"
    for f_name, f_func in func_dict.iteritems(): 
        #exec('{} = f_func'.format(f_name))
        print(f_func)
    #print('Globals:', g_var)
    #print('Locals:', l_var)
    g_var['np'] = np
    for i,v in l_var.iteritems(): g_var[i] = v
    exec('c_model = {}({})'.format(mdl_spawn_fcn_name, in_shape), g_var)
    c_model = g_var['c_model']
    for (lay_name, (lay_idx, lay_weights)) in out_weights.iteritems():
        assert lay_name == c_model.layers[lay_idx].name, "Names do not match {} != {}".format(lay_name, c_model.layers[lay_idx].name)
        c_weights = c_model.layers[lay_idx].get_weights()
        assert len(c_weights) == len(lay_weights), "{} layer weights must have same number".format(lay_name)
        c_shapes = [cw.shape for cw in c_weights]
        lay_weights = map(np.array,lay_weights) # make sure all of the arrays are numpy not python
        l_shapes = [lw.shape for lw in lay_weights]
        match_mat = all([cs == ls for cs,ls in zip(c_shapes,l_shapes)])
        mis_msg = "{} layer shapes must match {} != {}".format(lay_name, c_shapes, l_shapes )
        if not allow_mismatch:
            assert match_mat, mis_msg
        if match_mat:
            c_model.layers[lay_idx].set_weights(lay_weights)
        else:
            print(mis_msg)
    return ImageModel(c_model, preproc_fun)

import sys
import os
import h5py
import numpy as np
import json
from tempfile import NamedTemporaryFile

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
        with NamedTemporaryFile(suffix = '.h5') as c_weight_file:
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
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
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

    def save(self, out_dir, out_prefix = 'nn'):
        """Saves weights data (binary) and weights metadata (json)
        """
        model_filepath = '{}_model.json'.format(os.path.join(out_dir, out_prefix))
        with open(model_filepath,'w') as f:
            f.write(self.out_model.to_json())
        weights_filepath = '{}_weights.buf'.format(os.path.join(out_dir, out_prefix))
        with open(weights_filepath, mode='wb') as f:
            f.write(self.weights)
        metadata_filepath = '{}_metadata.json'.format(os.path.join(out_dir, out_prefix))
        with open(metadata_filepath, mode='w') as f:
            json.dump(self.metadata, f)
        return {'model_path': model_filepath, 'weights_path': weights_filepath, 'metadata_path': metadata_filepath}

    @staticmethod
    def saveModel(model, out_dir, out_prefix = 'nn'):
        """Saves everything in a single function
        """
        new_net = NetEncoder(model)
        new_net.serialize()
        return new_net.save(out_dir, out_prefix)