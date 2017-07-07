from skimage.measure import label, regionprops
from itertools import product
import numpy as np
from pyqae.utils import pprint # noinspection PyUnresolvedReferences
# we need this tests otherwise dicts get all uppity and randomly swap things
from collections import OrderedDict # noinspection PyUnresolvedReferences

__doc__ = """
This module is used for feeding image-based training data in 2/3D to a network.
The general format being followed is that of the Keras generators but they are bit more specialized for medical data and preprocessing steps
"""

_test_lab_img = np.expand_dims(np.expand_dims(np.stack([np.eye(3)] * 5, -1), -1), 0)
_test_ct_img = 1024 * _test_lab_img - 512
_test_pet_img = 5 * _test_lab_img

def label_trim(in_lab, trim_z, trim_x, trim_y):
    """
    Crop the edges for training (avoids convolution border issues and improves results on smaller tiles)
    :param in_lab:
    :param trim_z:
    :param trim_x:
    :param trim_y:
    :return:
    >>> _test_lab_img.shape
    (1, 3, 3, 5, 1)
    >>> label_trim(_test_lab_img, 1, 1, 2).shape
    (1, 1, 1, 1, 1)
    """
    if (trim_z == 0) & (trim_x == 0) & (trim_y == 0):
        return in_lab
    return in_lab[:, trim_z:-trim_z, trim_x:-trim_x, trim_y:-trim_y]


def label_zero_trim(in_lab, trim_z, trim_x, trim_y):
    """
    Zero-pad image for training
    :param in_lab:
    :param trim_z:
    :param trim_x:
    :param trim_y:
    :return:
    >>> _test_lab_img.shape
    (1, 3, 3, 5, 1)
    >>> label_zero_trim(_test_lab_img, 1, 1, 2).shape
    (1, 3, 3, 5, 1)
    >>> '%2.2f' % _test_lab_img.mean()
    '0.33'
    >>> '%2.2f' % label_zero_trim(_test_lab_img, 1, 1, 2).mean()
    '0.02'
    """
    new_lab = np.zeros_like(in_lab)
    new_lab[:, trim_z:-trim_z, trim_x:-trim_x, trim_y:-trim_y] = in_lab[:, trim_z:-trim_z, trim_x:-trim_x,
                                                                 trim_y:-trim_y]
    return new_lab


def random_region_generator(in_image_dict,  # type: Dict[str,np.ndarray]
                            out_image_dict,  # type: Dict[str,np.ndarray]
                            roi_dim,  # type: Tuple[int, int, int]
                            trim_dim  # type: Tuple[int, int, int]
                            ):
    # type: (...) ->  Generator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]
    """
    Randomly pick a region from the image using the numpy uniform sampler
    :param in_image_dict:
    :param out_image_dict:
    :param lab_image:
    :param roi_dim:
    :param trim_dim:
    :return:
    >>> in_data = OrderedDict(CT_Image = _test_ct_img, PET = _test_pet_img)
    >>> out_data = OrderedDict(Labels = _test_lab_img)
    >>> t_gen = random_region_generator(in_data, out_data, (2, 2, 2), (0, 0,0))
    >>> i_dict, o_dict = _get_first(t_gen)
    >>> for k,v in i_dict.items(): print(k,v.shape)
    CT_Image (1, 2, 2, 2, 1)
    PET (1, 2, 2, 2, 1)
    >>> for k,v in o_dict.items(): print(k,v.shape)
    Labels (1, 2, 2, 2, 1)
    >>> for key, val in i_dict.items(): print(key, val.mean())
    CT_Image 0.0
    PET 2.5
    >>> for key, val in o_dict.items(): print(key, val.mean())
    Labels 0.5
    >>> for key, val in i_dict.items(): pprint(val.squeeze())
    [[[ 512.  512.]
      [-512. -512.]]
    <BLANKLINE>
     [[-512. -512.]
      [ 512.  512.]]]
    [[[ 5.  5.]
      [ 0.  0.]]
    <BLANKLINE>
     [[ 0.  0.]
      [ 5.  5.]]]
    """
    roi_x, roi_y, roi_z = roi_dim
    trim_x, trim_y, trim_z = trim_dim
    while True:
        for _, ref_image in zip(range(1), in_image_dict.values()):
            c_img, z_p, x_p, y_p = [int(np.random.uniform(0, i - j)) for j, i in
                                    zip([1, roi_z, roi_x, roi_y], ref_image.shape)]
        gf = lambda x: x[c_img:(c_img + 1), z_p:(z_p + roi_z), x_p:(x_p + roi_x), y_p:(y_p + roi_y)]

        g_input = {lab_name: gf(lab_image)
                 for lab_name, lab_image in in_image_dict.items()}
        g_output = {lab_name: label_trim(gf(lab_image),trim_x=trim_x, trim_y=trim_y, trim_z=trim_z)
                 for lab_name, lab_image in out_image_dict.items()}
        yield g_input, g_output

def _get_first(x):
    for ix in x:
        return ix

def random_lesion_generator(in_image_dict,  # type: Dict[str,np.ndarray]
                            out_image_dict,  # type: Dict[str,np.ndarray]
                            roi_dim,  # type: Tuple[int, int, int]
                            trim_dim,  # type: Tuple[int, int, int]
                            rand_shift=0,
                            in_regs=None):
    # type: (...) -> Generator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]
    """
    Iterate through all of the lesions with rois sized in roi_dim
    :param in_image_dict:
    :param out_image_dict:
    :param roi_dim:
    :param trim_dim:
    :param rand_shift: the size of the random shift to be applied to the ROI
    :param in_regs:
    :return:
    >>> in_data = OrderedDict(CT_Image = _test_ct_img, PET = _test_pet_img)
    >>> out_data = OrderedDict(Labels = _test_lab_img)
    >>> t_gen = random_lesion_generator(in_data, out_data, (2, 2, 2), (0, 0,0))
    >>> i_dict, o_dict = _get_first(t_gen)
    >>> for k,v in i_dict.items(): print(k,v.shape)
    CT_Image (1, 2, 2, 2, 1)
    PET (1, 2, 2, 2, 1)
    >>> for k,v in o_dict.items(): print(k,v.shape)
    Labels (1, 2, 2, 2, 1)
    >>> for key, val in i_dict.items(): print(key, val.mean())
    CT_Image 0.0
    PET 2.5
    >>> for key, val in o_dict.items(): print(key, val.mean())
    Labels 0.5
    >>> for key, val in i_dict.items(): pprint(val.squeeze())
    [[[ 512.  512.]
      [-512. -512.]]
    <BLANKLINE>
     [[-512. -512.]
      [ 512.  512.]]]
    [[[ 5.  5.]
      [ 0.  0.]]
    <BLANKLINE>
     [[ 0.  0.]
      [ 5.  5.]]]
    """
    roi_x, roi_y, roi_z = roi_dim
    trim_x, trim_y, trim_z = trim_dim
    while True:
        if in_regs is None:
            for _, ref_image in zip(range(1), in_image_dict.values()):
                c_img, z_p, x_p, y_p = [int(np.random.uniform(0, i - j)) for
                                        j, i in
                                        zip([1, roi_z, roi_x, roi_y],
                                            ref_image.shape)]
            for _, ref_lab_image in zip(range(1), out_image_dict.values()):
                cur_regs = regionprops(label(np.sum(ref_lab_image[c_img], -1)))
        else:
            c_img, cur_regs = in_regs
        for c_reg in cur_regs:
            zmin, xmin, ymin, zmax, xmax, ymax = c_reg.bbox
            z_r = range(zmin, zmax, roi_z)
            x_r = range(xmin, xmax, roi_x)
            y_r = range(ymin, ymax, roi_y)
            for z_p, x_p, y_p in product(z_r, x_r, y_r):
                if rand_shift > 0:
                    z_p += int(np.random.uniform(-rand_shift, rand_shift))
                    x_p += int(np.random.uniform(-rand_shift, rand_shift))
                    y_p += int(np.random.uniform(-rand_shift, rand_shift))
                gf = lambda x: x[c_img:(c_img + 1), z_p:(z_p + roi_z), x_p:(x_p + roi_x), y_p:(y_p + roi_y)]
                g_input = {lab_name: gf(lab_image)
                           for lab_name, lab_image in in_image_dict.items()}
                g_output = {lab_name: label_trim(gf(lab_image), trim_x=trim_x,
                                                 trim_y=trim_y, trim_z=trim_z)
                            for lab_name, lab_image in out_image_dict.items()}
                yield g_input, g_output


def round_robin_gen(*gen_list):
    """
    Alternate between a list of different generators
    :param gen_list:
    :return:
    """
    while True:
        for c_gen in gen_list:
            yield next(c_gen)
