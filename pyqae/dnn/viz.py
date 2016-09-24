from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import skimage.transform
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.pyplot as plt # plotting
from skimage.io import imread # read in images
from skimage.segmentation import mark_boundaries # mark labels
from sklearn.metrics import roc_curve, auc # roc curve tools
from skimage.color import label2rgb
import numpy as np # linear algebra / matrices
from skimage.util.montage import montage2d

from scipy.stats import norm
from scipy import ndimage

from skimage.util.montage import montage2d
from skimage.io import imsave
import matplotlib.cm as cm

import keras.utils.visualize_util as vu
from IPython.display import SVG

def calc_montage_layer_view(td_seg_model, layer_id, img_in, border_padding = 1, 
                       scale_f = 2.0, cmap_fun = cm.RdBu, verbose = False):
    """
    Calculate a RGB representation of the image in the 3rd layer of the neural network
    """
    img_representation = layer_id.get_output_at(0)
    if verbose: print('input:',img_in.shape,'layer:',layer_id.name,'output:',img_representation)
    img_func = K.function([td_seg_model.input,  K.learning_phase()],[img_representation])
    
    rgb_outputs = img_func([img_in,0])[0] # training phase so elements are removed
    if verbose: print('out_shape',rgb_outputs.shape)
    _, _, l_wid, l_height = rgb_outputs.shape
    rgb8_outputs = rgb_outputs
    rgb8_outputs = (rgb8_outputs - np.mean(rgb8_outputs))/(scale_f*rgb8_outputs.std()) + 0.5
    rgb8_outputs = rgb8_outputs.clip(-1,1).astype(np.float32).reshape((-1,l_wid,l_height))
    if border_padding > 0:
        rgb8_outputs = np.pad(rgb8_outputs, border_padding, mode='constant')[border_padding:-border_padding]
    rgb_montage = montage2d(rgb8_outputs)
    if verbose: print('montage_shape',rgb_montage.shape)
    return cmap_fun(rgb_montage)

def show_full_network(cur_network, out_path = 'temp_network.svg'):
	"""
	Show a graph representation of the network
	"""
	dmodel = vu.model_to_dot(cur_network, show_shapes = True)
	dmodel.write_svg(out_path)
	return SVG(out_path)