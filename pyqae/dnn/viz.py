from __future__ import absolute_import, division, print_function

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
import keras.backend as K

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
	
def multichannel_softmax_image(p_img, cut_off = 0.5):
	"""
	Make a label image by calculating a multichannel softmax on a tensor
	"""
	return np.argmax(np.pad(p_img, ((0,0), (1,0), (0,0), (0,0)), 'constant', constant_values=cut_off),1)

def montage_rgb(arr_in, fill='mean', grid_shape=None, border_padding=0):
    """Create a 2-dimensional 'montage' from a 3-dimensional input array
    representing an ensemble of equally shaped 2-dimensional images.

    For example, ``montage2d(arr_in, fill)`` with the following `arr_in`

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    will return:

    +---+---+
    | 1 | 2 |
    +---+---+
    | 3 | * |
    +---+---+

    Where the '*' patch will be determined by the `fill` parameter.

    Parameters
    ----------
    arr_in : ndarray, shape=[n_images, height, width, n_channels]
        3-dimensional input array representing an ensemble of n_images
        of equal shape (i.e. [height, width, n_channels]).
    fill : float or 'mean', optional
        How to fill the 2-dimensional output array when sqrt(n_images)
        is not an integer. If 'mean' is chosen, then fill = arr_in.mean().
    grid_shape : tuple, optional
        The desired grid shape for the montage (tiles_y, tiles_x).
        The default aspect ratio is square.
    border_padding : int, optional
        The size of the spacing between the tiles to make the 
        boundaries of individual frames easier to see.

    Returns
    -------
    arr_out : ndarray, shape=[alpha * height, alpha * width, n_channels]
        Output array where 'alpha' has been determined automatically to
        fit (at least) the `n_images` in `arr_in`.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.montage import montage2d
    >>> arr_in = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    """
    
    assert arr_in.ndim == 4
    
    # -- add border padding, np.pad does all dimensions 
    # so we remove the padding from the first
    if border_padding > 0:
        arr_in = np.pad(arr_in, border_padding, mode='constant')[border_padding:-border_padding,:,:,border_padding:-border_padding]
    else:
        arr_in = arr_in.copy()
    
    n_images, height, width, n_channels = arr_in.shape


    # -- determine alpha
    if grid_shape:
        alpha_y, alpha_x = grid_shape
    else:
        alpha_y = alpha_x = int(np.ceil(np.sqrt(n_images)))

    # -- fill missing patches
    if fill == 'mean':
        fill = arr_in.mean()

    n_missing = int((alpha_y * alpha_x) - n_images)
    missing = (np.ones((n_missing, height, width, n_channels), dtype=arr_in.dtype) * fill).astype(arr_in.dtype)
    arr_out = np.vstack((arr_in, missing))

    # -- reshape to 2d montage, step by step
    arr_out = arr_out.reshape(alpha_y, alpha_x, height, width, n_channels)
    arr_out = arr_out.swapaxes(1, 2)
    arr_out = arr_out.reshape(alpha_y * height, alpha_x * width, n_channels)

    return arr_out