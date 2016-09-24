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
    def __init__(self, imodel):
        self.__rmodel = imodel
        self.input_size = imodel.get_input_shape_at(0)
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
        if prepend_shape:
            while len(in_image.shape)<len(self.input_size):
                in_image = np.expand_dims(in_image,0)
        assert len(in_image.shape) == len(self.input_size), "Image dimensions do not match model: {} != {}".format(in_image.shape, self.input_size)
        
        for i, (cur_size,mod_size) in enumerate(zip(in_image.shape, self.input_size)):
            if mod_size is not None: # batch dimension is none
                if pad_shape:
                    if cur_size < mod_size:
                        cur_size = mod_size ##TODO actually pad it with np.pad
                assert cur_size == mod_size, "Image at dimension at {} must match ({}, {})".format(i,in_image.shape, self.input_size)
        return self.__rmodel.predict(in_image,*args, **kwargs)


def write_msh_model(file_path, in_model, model_fcns, model_fcn_name = None, make_json = True):
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
    
    model_fcn_name: str
        The name of the function which can be called with the size to create a network
    
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
        weight_dict = {}
        for i, ilayer in enumerate(in_model.layers):
            weight_dict[ilayer.name] = [i,ilayer.get_weights()]
        marshal.dump(weight_dict,wf)
    if make_json:
        out_dict = {}
        out_dict['imp_code'] = STD_IMPORT_CODE
        out_dict['model_fcn_name'] = model_fcn_name
        out_dict['model_fcns'] = [(cfunc.func_name, inspect.getsourcelines(cfunc)) for cfunc in model_fcns]
        out_dict['weights'] = weight_dict
        with open('{}.json'.format(file_path),'w') as wf:
            json.dump(out_dict, wf, cls = NumpyAwareJSONEncoder)


def load_msh_model(file_path, weights_path):
    """
    Loads a model and weights from a given path
    """
    with open('{}.msh'.format(file_path),'rb') as rf:
        imp_code = marshal.load(rf)
        exec(imp_code)
        mdl_spawn_fcn_name = marshal.load(rf)
        assert type(mdl_spawn_fcn_name) is str, "Model name must be string -> {}".format(mdl_spawn_fcn_name)
        func_count = marshal.load(rf)
        for i in range(func_count):
            try:
                f_name = marshal.load(rf)
                code = marshal.load(rf)
                temp_func = types.FunctionType(code, globals(), f_name)
                exec('{} = temp_func'.format(f_name))
            except:
                print("Error reading function",i)
                break
        out_weights = marshal.load(rf)
        assert type(out_weights) is dict, "The weights must be stored in a dictionary"
    exec('c_model = %s' % (mdl_spawn_fcn_name,))
    for (lay_name, (lay_idx, lay_weights)) in out_weights.iteritems():
        c_lay_shape = c_model.layer[lay_idx].get_weights().shape
        assert c_lay_shape == lay_weights.shape, "Layer shapes must match {} != {}".format(c_lay_shape, lay_weights.shape)
        c_model.layer[lay_idx].set_weights(lay_weights)
    c_model.load_weights(weights_path)
    return c_model