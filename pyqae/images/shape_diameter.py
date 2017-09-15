__doc__ = """
A set of standard features we can calculate from images along with the 
appropriate tests. Currently we use them for a series of lesion diameter 
proxies
"""

from collections import OrderedDict

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import ball  # noinspection PyUnresolvedReferences

from pyqae.nd import change_resolution_array, autocrop
from pyqae.utils import pprint  # noinspection PyUnresolvedReferences

dummy_image = np.array(([0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 0, 0]))


def get_all_radii(in_vol, in_vox_size):
    """
    Calculates all of the radius metrics for a given volume
    :param in_vol:
    :param in_vox_size:
    :return:
    >>> for i,k in get_all_radii(ball(3), [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 6.0
    diameter_axial_edge_dist 6.0
    diameter_3d_edge_dist 6.0
    diameter_sphere 6.0
    >>> for i,k in get_all_radii(ball(3), [1,2,1]).items(): print(i,k//1)
    diameter_max_major_axial 12.0
    diameter_axial_edge_dist 7.0
    diameter_3d_edge_dist 6.0
    diameter_sphere 7.0
    >>> from skimage.morphology import cube, octahedron
    >>> for i,k in get_all_radii(cube(3), [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 3.0
    diameter_axial_edge_dist 7.0
    diameter_3d_edge_dist 8.0
    diameter_sphere 3.0
    >>> for i,k in get_all_radii(octahedron(3), [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 5.0
    diameter_axial_edge_dist 5.0
    diameter_3d_edge_dist 4.0
    diameter_sphere 4.0
    >>> bslice = np.expand_dims(ball(3)[ball(3).shape[0]//2],0)
    >>> for i,k in get_all_radii(bslice, [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 6.0
    diameter_axial_edge_dist 6.0
    diameter_3d_edge_dist 6.0
    diameter_sphere 3.0
    >>> for i,k in get_all_radii(np.zeros((3,3,3)), [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 0
    diameter_axial_edge_dist 0
    diameter_3d_edge_dist 0
    diameter_sphere -0.0
    >>> sp_image = np.zeros((3,3,3))
    >>> sp_image[1,1,1] = 1 # ensure it works on single point images / slices
    >>> for i,k in get_all_radii(sp_image, [1,1,1]).items(): print(i,k//1)
    diameter_max_major_axial 1
    diameter_axial_edge_dist 2.0
    diameter_3d_edge_dist 2.0
    diameter_sphere 1.0
    """
    out_vals = OrderedDict()
    out_vals['diameter_max_major_axial'] = get_2d_major_axis(in_vol,
                                                             in_vox_size)
    out_vals['diameter_axial_edge_dist'] = get_2d_max_edge_distance(in_vol,
                                                                    in_vox_size)
    out_vals['diameter_3d_edge_dist'] = get_3d_max_edge_distance(in_vol,
                                                                 in_vox_size)
    out_vals['diameter_sphere'] = 2 * np.power(np.sum(in_vol > 0) * np.prod(
        in_vox_size) * 3 / (4 * np.pi), 1 / 3.0)
    return out_vals


def get_2d_major_axis(in_vol, in_vox_size, axis=0):
    """
    Calculate the largest 2d major axis length of the object
    :param in_vol:
    :param in_vox_size:
    :param axis:
    :return:
    >>> get_2d_major_axis(ball(3), [1,1,1]) // 1
    6.0
    >>> get_2d_major_axis(ball(3), [1,2,1]) // 1
    12.0
    """
    ac_vol = autocrop(in_vol, 0)
    if np.prod(ac_vol.shape) == 0:
        return 0  # empty image

    if axis != 0: raise NotImplementedError(
        'Only axial radius is currently implemented')

    rs_vol, _ = change_resolution_array(ac_vol,
                                        old_vox_size=in_vox_size,
                                        new_vox_size=[1.0, 1.0, 1.0],
                                        order=2)

    return np.max([1] +  # if the image is only 1 pixel
                  [
                      # calculate max slice by slice
                      np.max([-1] + [
                          # in the event there are multiple regions in a slice
                          creg.major_axis_length for creg in
                          regionprops((in_slice > 0.5).astype(int))
                          # regionprops runs
                          # squeeze and doesnt work on 1d points
                      ])
                      for in_slice in rs_vol if
                      len(np.squeeze(in_slice).shape) > 1]
                  )


def get_2d_max_edge_distance(in_vol, in_vox_size, axis=0):
    """
    Calculate the largest 2D radius of a 3D object
    :param in_vol:
    :param in_vox_size:
    :return:
    >>> get_2d_max_edge_distance(ball(3), [1,1,1]) // 1
    6.0
    >>> get_2d_max_edge_distance(ball(3), [2,1,1]) // 1
    6.0
    >>> get_2d_max_edge_distance(ball(9), [1,1,1]) // 1
    18.0
    """
    # for in_idx in range(in_vol.shape[axis]):
    if axis != 0: raise NotImplementedError(
        'Only axial radius is currently implemented')
    ac_vol = autocrop(in_vol, 0)
    if np.prod(ac_vol.shape) == 0:
        return 0  # empty image

    return 2 * np.max([np.max(distance_transform_edt(in_slice,
                                                     sampling=in_vox_size[
                                                              1:]).ravel())
                       for in_slice in ac_vol])


def get_3d_max_edge_distance(in_vol, in_vox_size, resample=False):
    """
    Calculate the largest 3D radius of a 3D object
    :param in_vol:
    :param in_vox_size:
    :param resample: resample the volume rather than just inside the
    distance transform
    :return:
    >>> get_3d_max_edge_distance(ball(3), [1,1,1]) // 1
    6.0
    >>> get_3d_max_edge_distance(ball(3), [2,2,2]) // 1
    12.0
    >>> get_3d_max_edge_distance(ball(9), [1,1,1]) // 1
    18.0
    """
    ac_vol = autocrop(in_vol, 0)
    if np.prod(ac_vol.shape) == 0:
        return 0  # empty image

    if resample:
        rs_vol, _ = change_resolution_array(ac_vol,
                                            old_vox_size=in_vox_size,
                                            new_vox_size=[1.0, 1.0, 1.0],
                                            order=2)
        dmap = distance_transform_edt(rs_vol > 0.5)
    else:
        dmap = distance_transform_edt(ac_vol, sampling=in_vox_size)

    return 2 * np.max(dmap.ravel())


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
