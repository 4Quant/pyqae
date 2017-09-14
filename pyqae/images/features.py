__doc__ = """
A set of standard features we can calculate from images along with the 
appropriate tests
"""
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import ball  # noinspection PyUnresolvedReferences

from pyqae.nd import change_resolution_array
from pyqae.utils import pprint  # noinspection PyUnresolvedReferences

dummy_image = np.array(([0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 0, 0]))


def get_2d_radius(in_vol, in_vox_size):
    """
    Calculate the largest 2D radius of a 3D object
    :param in_vol:
    :param in_vox_size:
    :return:
    >>> get_2d_radius(ball(3), [1,1,1]) // 1
    3.0
    >>> get_2d_radius(ball(3), [2,1,1]) // 1
    3.0
    >>> get_2d_radius(ball(9), [1,1,1]) // 1
    9.0
    """
    return np.max([np.max(distance_transform_edt(in_slice,
                                                 sampling=in_vox_size[
                                                          1:]).ravel())
                   for in_slice in in_vol])


def get_3d_radius(in_vol, in_vox_size, resample=False):
    """
    Calculate the largest 3D radius of a 3D object
    :param in_vol:
    :param in_vox_size:
    :param resample: resample the volume rather than just inside the
    distance transform
    :return:
    >>> get_3d_radius(ball(3), [1,1,1]) // 1
    3.0
    >>> get_3d_radius(ball(3), [2,2,2]) // 1
    6.0
    >>> get_3d_radius(ball(9), [1,1,1]) // 1
    9.0
    """
    if resample:
        rs_vol, _ = change_resolution_array(in_vol,
                                            old_vox_size=in_vox_size,
                                            new_vox_size=[1.0, 1.0, 1.0],
                                            order=2)
        dmap = distance_transform_edt(rs_vol > 0.25)
    else:
        dmap = distance_transform_edt(in_vol, sampling=in_vox_size)

    return np.max(dmap.ravel())


def _simple_dmap_tests():
    """
    Some basic tests of the distance map
    :return:
    >>> pprint(distance_transform_edt(dummy_image))
    [[ 0.    1.    1.41  2.24  3.  ]
     [ 0.    0.    1.    2.    2.  ]
     [ 0.    1.    1.41  1.41  1.  ]
     [ 0.    1.    1.41  1.    0.  ]
     [ 0.    1.    1.    0.    0.  ]]
    >>> pprint(distance_transform_edt(dummy_image, sampling=[2,1]))
    [[ 0.    1.    2.    2.83  3.61]
     [ 0.    0.    1.    2.    3.  ]
     [ 0.    1.    2.    2.24  2.  ]
     [ 0.    1.    2.    1.    0.  ]
     [ 0.    1.    1.    0.    0.  ]]
    >>> edt, inds = distance_transform_edt(dummy_image, return_indices=True)
    >>> pprint(edt)
    [[ 0.    1.    1.41  2.24  3.  ]
     [ 0.    0.    1.    2.    2.  ]
     [ 0.    1.    1.41  1.41  1.  ]
     [ 0.    1.    1.41  1.    0.  ]
     [ 0.    1.    1.    0.    0.  ]]
    >>> pprint(inds)
    [[[0 0 1 1 3]
      [1 1 1 1 3]
      [2 2 1 3 3]
      [3 3 4 4 3]
      [4 4 4 4 4]]
    <BLANKLINE>
     [[0 0 1 1 4]
      [0 1 1 1 4]
      [0 0 1 4 4]
      [0 0 3 3 4]
      [0 0 3 3 4]]]
    >>> pprint(distance_transform_edt(ball(3), sampling=[1,2,1]))
    [[[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    1.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    1.    1.    1.    0.    0.  ]
      [ 0.    1.    1.    1.    1.    1.    0.  ]
      [ 0.    1.    1.    1.41  1.    1.    0.  ]
      [ 0.    1.    1.    1.    1.    1.    0.  ]
      [ 0.    0.    1.    1.    1.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    1.    1.41  2.    1.41  1.    0.  ]
      [ 0.    1.    2.    2.    2.    1.    0.  ]
      [ 0.    1.    2.    2.24  2.    1.    0.  ]
      [ 0.    1.    2.    2.    2.    1.    0.  ]
      [ 0.    1.    1.41  2.    1.41  1.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    1.    0.    0.    0.  ]
      [ 0.    1.    2.    2.24  2.    1.    0.  ]
      [ 0.    1.    2.    3.    2.    1.    0.  ]
      [ 1.    1.41  2.24  3.16  2.24  1.41  1.  ]
      [ 0.    1.    2.    3.    2.    1.    0.  ]
      [ 0.    1.    2.    2.24  2.    1.    0.  ]
      [ 0.    0.    0.    1.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    1.    1.41  2.    1.41  1.    0.  ]
      [ 0.    1.    2.    2.    2.    1.    0.  ]
      [ 0.    1.    2.    2.24  2.    1.    0.  ]
      [ 0.    1.    2.    2.    2.    1.    0.  ]
      [ 0.    1.    1.41  2.    1.41  1.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    1.    1.    1.    0.    0.  ]
      [ 0.    1.    1.    1.    1.    1.    0.  ]
      [ 0.    1.    1.    1.41  1.    1.    0.  ]
      [ 0.    1.    1.    1.    1.    1.    0.  ]
      [ 0.    0.    1.    1.    1.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    1.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.  ]]]
    """
    raise NotImplementedError("Just a test function")
