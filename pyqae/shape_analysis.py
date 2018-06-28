"""
Tools for analyzing the labels and data made in lungstage
"""
import warnings
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyqae.nd import autocrop
from pyqae.viz import draw_3d_labels
from skimage.measure import regionprops

try:
    # sklearn team renamed the core marching cubes and replaced it with something funkier
    from skimage.measure import marching_cubes_classic as marching_cubes
except ImportError:
    from skimage.measure import marching_cubes

from skimage.filters import gaussian
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fake_HTML = lambda x: x  # dummy function


def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """

    attributes_list = []

    for i, test_attribute in enumerate(dir(im_props[0])):

        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        try:
            if test_attribute[:1] != '_' and not \
                    isinstance(getattr(im_props[0], test_attribute), np.ndarray):
                attributes_list += [test_attribute]
        except Exception as e:
            warn("Not implemented: {} - {}".format(test_attribute, e), RuntimeWarning)

    return attributes_list


def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]

        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)


def region_analysis(cur_mask, cur_image):
    assert cur_mask.shape == cur_image.shape, "Mask and image must have same shape\n {} != {}".format(cur_mask.shape,
                                                                                                      cur_image.shape)
    return regionprops_to_df(regionprops(cur_mask, intensity_image=cur_image))
    

if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    import lungstage.shape_analysis

    doctest.testmod(lungstage.shape_analysis, verbose=True, optionflags=doctest.ELLIPSIS)
