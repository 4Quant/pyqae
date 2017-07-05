from warnings import warn

import numpy as np
import tensorflow as tf
from skimage.measure import regionprops

from pyqae.nd import meshgridnd_like, get_bbox, apply_bbox
from pyqae.utils import pprint  # noinspection PyUnresolvedReferences

__doc__ = """
A set of Tensorflow-based layers and operations for including in models
allowing meaningful spatial and medical information to be included in images
"""


def _setup_and_test(in_func, *in_arrs, is_list=False, round=False):
    """
    For setting up a simple graph and testing it
    :param in_func:
    :param in_arrs:
    :param is_list:
    :return:
    """
    with tf.Graph().as_default() as g:
        in_vals = [tf.placeholder(dtype=tf.float32, shape=in_arr.shape) for
                   in_arr in in_arrs]
        out_val = in_func(*in_vals)
        if not is_list:
            print('setup_net', [in_arr.shape for in_arr in in_arrs],
                  out_val.shape)
            out_list = [out_val]
        else:
            out_list = list(out_val)
    with tf.Session(graph=g) as c_sess:
        sess_out = c_sess.run(fetches=out_list,
                              feed_dict={in_val: in_arr
                                         for in_val, in_arr in
                                         zip(in_vals, in_arrs)})
        if is_list:
            o_val = sess_out
        else:
            o_val = sess_out[0]
        if round:
            return (np.array(o_val) * 100).astype(int) / 100
        else:
            return o_val


_simple_dist_img = np.stack([np.eye(3), 0.5 * np.ones((3, 3))], -1)


def add_vgrid_tf(in_layer,
                 x_cent,
                 y_cent,
                 x_wid,
                 y_wid,
                 z_cent,
                 append=True):
    """
    Adds spatial grids to images for making segmentation easier
    The add_vgrid_tf adds a grid based on a sample dependent list of
    coordinates
    >>> test_img = np.expand_dims(np.expand_dims(np.eye(3), 0), -1)
    >>> vec_fun = lambda x: np.array([x], dtype = np.float32).reshape((1,1))
    >>> t_cent = vec_fun(0)
    >>> t_wid = vec_fun(1.0)
    >>> out_img = _setup_and_test(add_vgrid_tf, test_img, t_cent, t_cent, t_wid, t_wid, t_cent)
    setup_net [(1, 3, 3, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)] (1, 3, 3, 4)
    >>> out_img.shape
    (1, 3, 3, 4)
    >>> out_img[0]
    array([[[ 1., -1., -1.,  0.],
            [ 0., -1.,  0.,  0.],
            [ 0., -1.,  1.,  0.]],
    <BLANKLINE>
           [[ 0.,  0., -1.,  0.],
            [ 1.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.]],
    <BLANKLINE>
           [[ 0.,  1., -1.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 1.,  1.,  1.,  0.]]], dtype=float32)
    """
    with tf.variable_scope('add_vgrid'):
        xg_wid = tf.shape(in_layer)[1]
        yg_wid = tf.shape(in_layer)[2]

        xx, yy, zz = tf.meshgrid(tf.linspace(-1.0, 1.0, xg_wid),
                                 tf.linspace(-1.0, 1.0, yg_wid),
                                 tf.linspace(0.0, 0.0, 1),
                                 indexing='ij')

        z_wid = x_wid  # it is multiplied by 0 anyways
        xx = tf.reshape(xx, (xg_wid, yg_wid, 1))
        yy = tf.reshape(yy, (xg_wid, yg_wid, 1))
        zz = tf.reshape(zz, (xg_wid, yg_wid, 1))
        fix_dim = lambda x: tf.expand_dims(tf.transpose(x, [2, 0, 1]), -1)
        xx_long = fix_dim(xx * x_wid + x_cent)
        yy_long = fix_dim(yy * y_wid + y_cent)
        zz_long = fix_dim(zz * z_wid + z_cent)

        txy_vec = tf.concat([xx_long, yy_long, zz_long], -1)
        if append:
            return tf.concat([in_layer, txy_vec], -1)
        else:
            return txy_vec


def add_simple_grid_tf(in_layer,  # type: tf.Tensor
                       x_cent=0.0,  # type: tf.Tensor
                       y_cent=0.0,  # type: tf.Tensor
                       x_wid=1.0,  # type: tf.Tensor
                       y_wid=1.0,  # type: tf.Tensor
                       z_cent=None  # type: Optional[tf.Tensor]
                       ):
    # type: (...) -> tf.Tensor
    """
    Adds spatial grids to images for making segmentation easier
    :param in_layer: the base image to use for x,y dimensions
    :param x_cent: the x mid coordinate
    :param y_cent: the y mid coordinate
    :param x_wid: the width in x (pixel spacing)
    :param y_wid: the width in y (pixel spacing)
    :param z_cent: the center location in z
    :return:
    """
    with tf.variable_scope('add_grid'):
        batch_size = tf.shape(in_layer)[0]
        xg_wid = tf.shape(in_layer)[1]
        yg_wid = tf.shape(in_layer)[2]
        x_min = x_cent - x_wid
        x_max = x_cent + x_wid
        y_min = y_cent - y_wid
        y_max = y_cent + y_wid

        if z_cent is None:
            xx, yy = tf.meshgrid(tf.linspace(x_min, x_max, xg_wid),
                                 tf.linspace(y_min, y_max, yg_wid),
                                 indexing='ij')
        else:
            xx, yy, zz = tf.meshgrid(tf.linspace(x_min, x_max, xg_wid),
                                     tf.linspace(y_min, y_max, yg_wid),
                                     tf.linspace(z_cent, z_cent, 1),
                                     indexing='ij')

        xx = tf.reshape(xx, (xg_wid, yg_wid, 1))
        yy = tf.reshape(yy, (xg_wid, yg_wid, 1))
        if z_cent is None:
            xy_vec = tf.expand_dims(tf.concat([xx, yy], -1), 0)
        else:
            zz = tf.reshape(zz, (xg_wid, yg_wid, 1))
            xy_vec = tf.expand_dims(tf.concat([xx, yy, zz], -1), 0)
        txy_vec = tf.tile(xy_vec, [batch_size, 1, 1, 1])
        return tf.concat([in_layer, txy_vec], -1)


def add_com_grid_3d_tf(in_layer,
                       layer_concat=False,
                       as_r_vec=False
                       ):
    # type: (tf.Tensor, bool, bool) -> tf.Tensor
    """
    Adds spatial grids to images for making segmentation easier
    This particular example utilizes the image-weighted center of mass by
    summing the input layer across the channels.
    It also normalizes the grid by using the weighted standard deviation

    :param in_layer:
    :param layer_concat:
    :return:
    >>> _testimg = np.ones((5, 4, 3, 2, 1))
    >>> out_img = _setup_and_test(add_com_grid_3d_tf, _testimg)
    setup_net [(5, 4, 3, 2, 1)] (5, ?, ?, ?, 3)
    >>> out_img.shape
    (5, 4, 3, 2, 3)
    >>> pprint(out_img[0, :, 0, 0, 0])
    [-1.34 -0.45  0.45  1.34]
    >>> pprint(out_img[0, 0, :, 0, 1])
    [-1.22  0.    1.22]
    >>> pprint(out_img[0, 0, 0, :, 2])
    [-1.  1.]
    >>> out_img = _setup_and_test(lambda x: add_com_grid_3d_tf(x, as_r_vec=True), _testimg)
    setup_net [(5, 4, 3, 2, 1)] (5, ?, ?, ?, 1)
    >>> out_img.shape
    (5, 4, 3, 2, 1)
    >>> pprint(out_img[0, :, 0, 0, 0])
    [ 2.07  1.64  1.64  2.07]
    """
    with tf.variable_scope('com_grid_op'):
        with tf.variable_scope('initialize'):
            batch_size = tf.shape(in_layer)[0]
            xg_wid = tf.shape(in_layer)[1]
            yg_wid = tf.shape(in_layer)[2]
            zg_wid = tf.shape(in_layer)[3]

            mask_sum = tf.reduce_sum(in_layer, 4)
        with tf.variable_scope('setup_com'):

            _, xx, yy, zz = tf.meshgrid(
                tf.linspace(0., 0., batch_size),
                tf.linspace(-1., 1., xg_wid),
                tf.linspace(-1., 1., yg_wid),
                tf.linspace(-1., 1., zg_wid),
                indexing='ij')

        with tf.variable_scope('calc_com'):
            svar_list = [xx, yy, zz]
            sm_list = [
                tf.reduce_sum(c_var * mask_sum, [1, 2, 3]) * 1 / tf.reduce_sum(
                    mask_sum, [1, 2, 3]) for c_var in svar_list]
            # wstd is np.sqrt(np.sum(w_y*np.square(t_x))/np.sum(w_y)-np.square(np.sum(w_y*t_x)/np.sum(w_y)))
            sd_list = [
                tf.sqrt(tf.reduce_sum(mask_sum * tf.square(c_var), [1, 2, 3]) /
                        tf.reduce_sum(mask_sum, [1, 2, 3]) - tf.square(m_var))
                for m_var, c_var in zip(sm_list, svar_list)
            ]
            expand_op = lambda iv: tf.expand_dims(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(iv, -1), -1), -1),
                -1)
            tile_op = lambda iv: tf.tile(iv, [1, xg_wid, yg_wid, zg_wid, 1])
            res_op = lambda iv: tile_op(expand_op(iv))

            sm_matlist = [res_op(c_var) for c_var in sm_list]
            sd_matlist = [res_op(c_var) for c_var in sd_list]

        with tf.variable_scope('make_grid'):
            out_var = [(tf.reshape(c_var, (
                batch_size, xg_wid, yg_wid, zg_wid, 1)) - c_sm) / c_sd
                       for c_var, c_sm, c_sd in zip(svar_list, sm_matlist,
                                                    sd_matlist)]

            xy_vec = tf.concat(out_var, -1)
            if as_r_vec:
                xy_vec = tf.expand_dims(
                    tf.sqrt(tf.reduce_sum(tf.square(xy_vec), -1)),
                    -1)

        if layer_concat:
            return tf.concat([in_layer, xy_vec], -1)
        else:
            return xy_vec


def add_com_grid_2d_tf(in_layer,
                       layer_concat=False,
                       as_r_vec=False,
                       r_scale=1.0
                       ):
    # type: (tf.Tensor, bool, bool, float) -> tf.Tensor
    """
    Adds spatial grids to images for making segmentation easier
    This particular example utilizes the image-weighted center of mass by
    summing the input layer across the channels.
    It also normalizes the grid by using the weighted standard deviation

    :param in_layer:
    :param layer_concat:
    :param as_r_vec: return the r vector
    :param r_scale: the value to rescale the standard deviation by when 
    creating r
    :return:
    >>> _testimg = np.ones((4, 3, 2, 1))
    >>> out_img = _setup_and_test(add_com_grid_2d_tf, _testimg)
    setup_net [(4, 3, 2, 1)] (4, ?, ?, 2)
    >>> out_img.shape
    (4, 3, 2, 2)
    >>> pprint(out_img[0, :, 0, 0])
    [-1.22  0.    1.22]
    >>> pprint(out_img[0, 0, :, 1])
    [-1.  1.]
    >>> out_img = _setup_and_test(lambda x: add_com_grid_2d_tf(x, as_r_vec=True), _testimg)
    setup_net [(4, 3, 2, 1)] (4, ?, ?, 1)
    >>> out_img.shape
    (4, 3, 2, 1)
    >>> pprint(out_img[0, :, 0, 0])
    [ 1.58  1.    1.58]
    """
    with tf.variable_scope('com_grid_op_2d'):
        with tf.variable_scope('initialize'):
            batch_size = tf.shape(in_layer)[0]
            xg_wid = tf.shape(in_layer)[1]
            yg_wid = tf.shape(in_layer)[2]

            mask_sum = tf.reduce_sum(in_layer, 3)
        with tf.variable_scope('setup_com'):

            _, xx, yy = tf.meshgrid(
                tf.linspace(0., 0., batch_size),
                tf.linspace(-1., 1., xg_wid),
                tf.linspace(-1., 1., yg_wid),
                indexing='ij')

        with tf.variable_scope('calc_com'):
            svar_list = [xx, yy]
            sm_list = [
                tf.reduce_sum(c_var * mask_sum, [1, 2]) * 1 / tf.reduce_sum(
                    mask_sum, [1, 2]) for c_var in svar_list]
            # wstd is np.sqrt(np.sum(w_y*np.square(t_x))/np.sum(w_y)-np.square(np.sum(w_y*t_x)/np.sum(w_y)))
            sd_list = [
                tf.sqrt(tf.reduce_sum(mask_sum * tf.square(c_var), [1, 2]) /
                        tf.reduce_sum(mask_sum, [1, 2]) - tf.square(m_var))
                for m_var, c_var in zip(sm_list, svar_list)
            ]
            expand_op = lambda iv: tf.expand_dims(tf.expand_dims(
                tf.expand_dims(iv, -1), -1), -1)
            tile_op = lambda iv: tf.tile(iv, [1, xg_wid, yg_wid, 1])
            res_op = lambda iv: tile_op(expand_op(iv))

            sm_matlist = [res_op(c_var) for c_var in sm_list]
            sd_matlist = [res_op(c_var) for c_var in sd_list]

        with tf.variable_scope('make_grid'):
            out_var = [(tf.reshape(c_var, (
                batch_size, xg_wid, yg_wid, 1)) - c_sm) / (r_scale * c_sd)
                       for c_var, c_sm, c_sd in zip(svar_list, sm_matlist,
                                                    sd_matlist)]

            xy_vec = tf.concat(out_var, -1)
            if as_r_vec:
                xy_vec = tf.expand_dims(
                    tf.sqrt(tf.reduce_sum(tf.square(xy_vec), -1)),
                    -1)

        if layer_concat:
            return tf.concat([in_layer, xy_vec], -1)
        else:
            return xy_vec


def spatial_gradient_tf(in_img):
    # type: (tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    Calculate the spatial gradient in x,y,z using tensorflow
    The channel dimension is completely ignored and the batches are kept
    consistent.
    :param in_img: a 5d tensor sized as batch, x, y, z, channel
    :return:
    NOTE:: the doctests are written to only test the main region, due to
    boundary issues the edges are different (not massively) between
    np.gradient and this function, eventually a better edge scaling should
    be implemented
    >>> _testimg = np.ones((4, 4, 4))
    >>> _testimg = np.sum(np.power(np.stack(meshgridnd_like(_testimg),-1),2),-1)
    >>> _testimg = np.expand_dims(np.expand_dims(_testimg,0),-1)
    >>> dx, dy, dz = _setup_and_test(spatial_gradient_tf, _testimg, is_list=True)
    >>> dx.shape
    (1, 4, 4, 4, 1)
    >>> dy.shape
    (1, 4, 4, 4, 1)
    >>> dz.shape
    (1, 4, 4, 4, 1)
    >>> ndx, ndy, ndz = np.gradient(_testimg[0,:,:,:,0])
    >>> [(a,b) for a,b in zip(ndx[1:-1,0,0],dx[0,1:-1,0,0,0])]
    [(2.0, 2.0), (4.0, 4.0)]
    >>> [(a,b) for a,b in zip(ndy[0,1:-1,0],dy[0,0,1:-1,0,0])]
    [(2.0, 2.0), (4.0, 4.0)]
    >>> [(a,b) for a,b in zip(ndz[0,0,1:-1],dz[0,0,0,1:-1,0])]
    [(2.0, 2.0), (4.0, 4.0)]
    >>> np.sum(ndx-dx[0,:,:,:,0],(1,2))
    array([  8.,   0.,   0.,  40.])
    >>> np.sum(ndy-dy[0,:,:,:,0], (0,2))
    array([  8.,   0.,   0.,  40.])
    >>> np.sum(ndz-dz[0,:,:,:,0], (0,1))
    array([  8.,   0.,   0.,  40.])
    """
    with tf.variable_scope('spatial_gradient'):
        pad_r = tf.pad(in_img, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                       "SYMMETRIC")
        dx_img = pad_r[:, 2:, 1:-1, 1:-1, :] - pad_r[:, 0:-2, 1:-1, 1:-1, :]
        dy_img = pad_r[:, 1:-1, 2:, 1:-1, :] - pad_r[:, 1:-1, 0:-2, 1:-1, :]
        dz_img = pad_r[:, 1:-1, 1:-1, 2:, :] - pad_r[:, 1:-1, 1:-1, 0:-2, :]
        return (0.5 * dx_img, 0.5 * dy_img, 0.5 * dz_img)


def spatial_gradient_2d_tf(in_img):
    # type: (tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    Calculate the 2d spatial gradient in x,y using tensorflow
    The channel dimension is completely ignored and the batches are kept
    consistent.
    :param in_img: a 4d tensor sized as batch, x, y, channel
    :return:
    NOTE:: the doctests are written to only test the main region, due to
    boundary issues the edges are different (not massively) between
    np.gradient and this function, eventually a better edge scaling should
    be implemented
    >>> _testimg = np.ones((4, 4))
    >>> _testimg = np.sum(np.power(np.stack(meshgridnd_like(_testimg),-1),2),-1)
    >>> _testimg = np.expand_dims(np.expand_dims(_testimg,0),-1)
    >>> dx, dy = _setup_and_test(spatial_gradient_2d_tf, _testimg, is_list=True)
    >>> dx.shape, dy.shape
    ((1, 4, 4, 1), (1, 4, 4, 1))
    >>> ndx, ndy = np.gradient(_testimg[0,:,:,0])
    >>> (ndx.shape, ndy.shape)
    ((4, 4), (4, 4))
    >>> [(a,b) for a,b in zip(ndx[:,0],dx[0,1:-1,0,0])]
    [(1.0, 2.0), (2.0, 4.0)]
    >>> [(a,b) for a,b in zip(ndy[0,:],dy[0,0,1:-1,0])]
    [(1.0, 2.0), (2.0, 4.0)]
    >>> np.sum(ndx-dx[0,:,:,0],(1))
    array([  2.,   0.,   0.,  10.])
    >>> np.sum(ndy-dy[0,:,:,0], (0))
    array([  2.,   0.,   0.,  10.])
    """
    with tf.variable_scope('spatial_gradient_2d'):
        pad_r = tf.pad(in_img, [[0, 0], [1, 1], [1, 1], [0, 0]],
                       "SYMMETRIC")
        dx_img = pad_r[:, 2:, 1:-1, :] - pad_r[:, 0:-2, 1:-1, :]
        dy_img = pad_r[:, 1:-1, 2:, :] - pad_r[:, 1:-1, 0:-2, :]
        return (0.5 * dx_img, 0.5 * dy_img)


def phi_coord_3d_tf(r_img, z_rad=0.0,
                    include_r=False,
                    include_ir=False,
                    ir_smooth=1e-2):
    # type: (tf.Tensor, float, bool, bool) -> tf.Tensor
    """
    Calculate the phi coordinates for tensors using a single step
    derivatives and arc-sin to create smooth functions
    :param r_img:
    :param z_rad:
    :param include_r: include the r axis
    :param include_ir: include the inverted r_axis (+1e-1)
    :return:
    >>> _testimg = np.ones((4, 3, 2))
    >>> from scipy.ndimage.morphology import distance_transform_edt
    >>> _testimg = distance_transform_edt(_testimg)
    >>> _testimg = np.expand_dims(np.expand_dims(_testimg,0),-1)
    >>> out_img = _setup_and_test(phi_coord_3d_tf, _testimg)
    setup_net [(1, 4, 3, 2, 1)] (1, 4, 3, 2, 3)
    >>> out_img.shape
    (1, 4, 3, 2, 3)
    >>> pprint(out_img[0, :, 0, 0, 0])
    [ 0.33  0.45  0.46  0.45]
    >>> ntimg = np.expand_dims(np.expand_dims(_simple_dist_img,0),-1)
    >>> oimg = _setup_and_test(phi_coord_3d_tf, ntimg)[0]
    setup_net [(1, 3, 3, 2, 1)] (1, 3, 3, 2, 3)
    >>> pprint(oimg[:, 0 , 0, 0])
    [-0.23 -0.    0.  ]
    >>> pprint(oimg[:, 1 , 0, 1])
    [-0.  0.  0.]
    >>> pprint(oimg[:, 2 , 0, 2])
    [ 0.    0.   -0.11]
    >>> ffunc = lambda x: phi_coord_3d_tf(x, 0, True, True)
    >>> oimg = _setup_and_test(ffunc, ntimg)[0]
    setup_net [(1, 3, 3, 2, 1)] (1, 3, 3, 2, 5)
    >>> pprint(oimg[:, 0 , 0, 0])
    [ 1.  0.  0.]
    >>> pprint(oimg[:, 1 , 0, 1])
    [ 100.      0.99  100.  ]
    >>> pprint(oimg[:, 2 , 0, 2])
    [ 0.    0.    0.23]
    """
    with tf.variable_scope('phi_coord_3d'):
        (dx_img, dy_img, dz_img) = spatial_gradient_tf(r_img)
        dr_img = tf.sqrt(
            tf.square(dx_img) + tf.square(dy_img) + tf.square(dz_img))
        dr_img = tf.clip_by_value(dr_img, ir_smooth, tf.float32.max)
        mask_img = tf.cast(r_img > z_rad, tf.float32)
        safe_asin = lambda x: tf.asin(tf.clip_by_value(x, -1, 1))
        dphi_a_img = safe_asin(dx_img / dr_img) / np.pi * mask_img
        dphi_b_img = (safe_asin(dy_img / dr_img)) / np.pi * mask_img
        dphi_c_img = (safe_asin(dz_img / dr_img)) / np.pi * mask_img
        out_vec = [dphi_a_img, dphi_b_img, dphi_c_img]
        if include_ir:
            out_vec = [1 / (ir_smooth + r_img)] + out_vec
        if include_r:
            out_vec = [r_img] + out_vec
        return tf.concat(out_vec, -1)


def phi_coord_2d_tf(r_img,
                    z_rad=0.0,
                    include_r=False,
                    include_ir=False,
                    ir_smooth=1e-2):
    # type: (tf.Tensor, float, bool, bool) -> tf.Tensor
    """
    Calculate the phi coordinates for tensors using a single step
    derivatives and arc-sin to create smooth functions for 2d cases
    :param r_img:
    :param z_rad:
    :param include_r: include the r axis
    :param include_ir: include the inverted r_axis (+1e-1)
    :return:
    >>> _testimg = np.ones((4, 3))
    >>> from scipy.ndimage.morphology import distance_transform_edt
    >>> _testimg = distance_transform_edt(_testimg)
    >>> _testimg = np.expand_dims(np.expand_dims(_testimg,0),-1)
    >>> out_img = _setup_and_test(phi_coord_2d_tf, _testimg)
    setup_net [(1, 4, 3, 1)] (1, 4, 3, 2)
    >>> out_img.shape
    (1, 4, 3, 2)
    >>> pprint(out_img[0, :, 0, 0])
    [ 0.38  0.46  0.47  0.46]
    >>> ntimg = np.expand_dims(np.expand_dims(_simple_dist_img[:,:,0],0),-1)
    >>> oimg = _setup_and_test(phi_coord_2d_tf, ntimg)[0]
    setup_net [(1, 3, 3, 1)] (1, 3, 3, 2)
    >>> pprint(oimg[:, 0 , 0])
    [-0.25 -0.    0.  ]
    >>> pprint(oimg[:, 1 , 1])
    [-0.  0.  0.]
    >>> ffunc = lambda x: phi_coord_2d_tf(x, 0, True, True)
    >>> oimg = _setup_and_test(ffunc, ntimg)[0]
    setup_net [(1, 3, 3, 1)] (1, 3, 3, 4)
    >>> pprint(oimg[:, 0 , 0])
    [ 1.  0.  0.]
    >>> pprint(oimg[:, 1 , 1])
    [ 100.      0.99  100.  ]
    >>> pprint(oimg[:, 2 , 2])
    [ 0.    0.    0.25]
    """
    with tf.variable_scope('phi_coord_2d'):
        (dx_img, dy_img) = spatial_gradient_2d_tf(r_img)
        dr_img = tf.sqrt(
            tf.square(dx_img) + tf.square(dy_img))
        dr_img = tf.clip_by_value(dr_img, ir_smooth, tf.float32.max)
        mask_img = tf.cast(r_img > z_rad, tf.float32)
        safe_asin = lambda x: tf.asin(tf.clip_by_value(x, -1, 1))
        dphi_a_img = safe_asin(dx_img / dr_img) / np.pi * mask_img
        dphi_b_img = safe_asin(dy_img / dr_img) / np.pi * mask_img
        out_vec = [dphi_a_img, dphi_b_img]
        if include_ir:
            out_vec = [1 / (ir_smooth + r_img)] + out_vec
        if include_r:
            out_vec = [r_img] + out_vec
        return tf.concat(out_vec, -1)


def add_com_phi_grid_3d_tf(in_layer,
                           layer_concat=False,
                           z_rad=0.0,
                           include_r=False,
                           include_ir=False
                           ):
    # type: (tf.Tensor, bool, float, bool, bool) -> tf.Tensor
    """
    Adds spatial phi grids to 3d images for making segmentation easier
    This particular example utilizes the image-weighted center of mass by
    summing the input layer across the channels

    :param in_layer:
    :param layer_concat:
    :param z_rad: minimum radius to include
    :param include_r: include the radius channel
    :return:
    >>> _testimg = np.ones((5, 4, 3, 2, 1))
    >>> out_img = _setup_and_test(add_com_phi_grid_3d_tf, _testimg)
    setup_net [(5, 4, 3, 2, 1)] (5, ?, ?, ?, 3)
    >>> out_img.shape
    (5, 4, 3, 2, 3)
    >>> pprint(out_img[0, :, 0, 0, 0])
    [-0.26 -0.21  0.21  0.26]
    >>> pprint(out_img[0, 0, :, 0, 1])
    [-0.24  0.    0.24]
    >>> pprint(out_img[0, 0, 0, :, 2])
    [ 0.  0.]
    """
    with tf.variable_scope('add_com_phi_grid_3d'):
        r_vec = add_com_grid_3d_tf(in_layer, layer_concat=False, as_r_vec=True)
        phi_out = phi_coord_3d_tf(r_vec, z_rad=z_rad, include_r=include_r,
                                  include_ir=include_ir)
        if layer_concat:
            return tf.concat([in_layer, phi_out], -1)
        else:
            return phi_out


def add_com_phi_grid_2d_tf(in_layer,
                           layer_concat=False,
                           z_rad=0.0,
                           include_r=False,
                           include_ir=False,
                           r_scale=1.0
                           ):
    # type: (tf.Tensor, bool, float, bool, bool) -> tf.Tensor
    """
    Adds spatial phi grids to 2d images for making segmentation easier
    This particular example utilizes the image-weighted center of mass by
    summing the input layer across the channels

    :param in_layer:
    :param layer_concat:
    :param z_rad: minimum radius to include
    :param include_r: include the radius channel
    :param r_scale: the scale factor for the r axis
    :return:
    >>> _testimg = np.ones((4, 3, 2, 1))
    >>> out_img = _setup_and_test(add_com_phi_grid_2d_tf, _testimg)
    setup_net [(4, 3, 2, 1)] (4, ?, ?, 2)
    >>> out_img.shape
    (4, 3, 2, 2)
    >>> pprint(out_img[0, :, 0, 0])
    [-0.5  0.   0.5]
    >>> pprint(out_img[0, 0, :, 1])
    [ 0.  0.]
    """
    with tf.variable_scope('add_com_phi_grid_2d'):
        r_vec = add_com_grid_2d_tf(in_layer,
                                   layer_concat=False,
                                   as_r_vec=True,
                                   r_scale=r_scale)
        phi_out = phi_coord_2d_tf(r_vec, z_rad=z_rad,
                                  include_r=include_r,
                                  include_ir=include_ir)
        if layer_concat:
            return tf.concat([in_layer, phi_out], -1)
        else:
            return phi_out


def obj_to_phi_np(seg_img, z_rad=0):
    # type: (np.ndarray, float) -> np.ndarray
    """
    Create a phi mask from a given object
    :param seg_img:
    :param z_rad:
    :return:
    """
    c_reg = regionprops((seg_img > 0).astype(int))[0]
    return generate_phi_coord_np(seg_img,
                                 centroid=c_reg.centroid,
                                 std_xyz=[1, 1, 1],
                                 zrad=z_rad)


def generate_phi_coord_np(seg_img, centroid, std_xyz=[1, 1, 1], z_rad=0):
    # type: (np.ndarray, Tuple[float, float, float], float) -> np.ndarray
    """
    Create the phi coordinate system
    :param seg_img:
    :param centroid:
    :param z_rad:
    :return:
    """
    xx, yy, zz = meshgridnd_like(seg_img)
    r_img = np.sqrt(np.power((xx - centroid[0]) / std_xyz[0], 2) +
                    np.power((yy - centroid[1]) / std_xyz[1], 2) +
                    np.power((zz - centroid[2]) / std_xyz[2], 2))
    return phi_coord_np(r_img, z_rad=z_rad)


def phi_coord_np(r_img, z_rad):
    # type: (np.ndarray, float) -> np.ndarray
    """
    Calculate the phi coordinates using numpy
    :param r_img:
    :param z_rad:
    :return:
    >>> _simple_dist_img.shape
    (3, 3, 2)
    >>> oimg = phi_coord_np(_simple_dist_img,0)
    >>> oimg.shape
    (3, 3, 2, 3)
    >>> oimg[:, 0 , 0, 0]
    array([-0.23227953, -0.        ,  0.        ])
    >>> oimg[:, 1 , 0, 1]
    array([-0.,  0.,  0.])
    >>> oimg[:, 2 , 0, 2]
    array([ 0.        ,  0.        , -0.10817345])
    """
    dx_img, dy_img, dz_img = np.gradient(r_img)
    dr_img = np.sqrt(
        np.power(dx_img, 2) + np.power(dy_img, 2) + np.power(dz_img, 2))
    dphi_a_img = np.arcsin(dx_img / dr_img) / np.pi * (r_img > z_rad)
    dphi_b_img = (np.arcsin(dy_img / dr_img)) / np.pi * (r_img > z_rad)
    dphi_c_img = (np.arcsin(dz_img / dr_img)) / np.pi * (r_img > z_rad)
    return np.stack([dphi_a_img, dphi_b_img, dphi_c_img], -1)


def __compare_numpy_and_tf():
    """
    A series of functions for comparing tensorflow to numpy output and
    making sure the differences are small enough to be tolerated
    :return:
    >>> s_range = np.linspace(-1,1, 10)
    >>> _setup_and_test(lambda x: tf.cast(x>0, tf.float32), s_range)
    setup_net [(10,)] (10,)
    array([ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.], dtype=float32)
    >>> np_asin = np.arcsin(s_range)
    >>> tf_asin = _setup_and_test(tf.asin, s_range)
    setup_net [(10,)] (10,)
    >>> (tf_asin - np_asin)/np_asin
    array([ -4.80634556e-08,   1.35596255e-08,   1.19606977e-07,
             4.65342369e-08,   1.46818079e-09,   1.46818079e-09,
             4.65342372e-08,   1.19606977e-07,   1.35596256e-08,
            -4.80634556e-08])
    >>> oimg_np = phi_coord_np(_simple_dist_img,0)
    >>> ntimg = np.expand_dims(np.expand_dims(_simple_dist_img,0),-1)
    >>> oimg_tf = _setup_and_test(phi_coord_3d_tf, ntimg)
    setup_net [(1, 3, 3, 2, 1)] (1, 3, 3, 2, 3)
    >>> pprint(np.abs(oimg_tf[0]-oimg_np)/(np.abs(oimg_np)+.1))
    [[[[  8.07e-08   8.07e-08   5.67e-08]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  0.00e+00   0.00e+00   4.97e-08]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  0.00e+00   0.00e+00   0.00e+00]
       [  0.00e+00   0.00e+00   4.97e-08]]
    <BLANKLINE>
      [[  9.00e-09   9.00e-09   2.09e-08]
       [  0.00e+00   0.00e+00   4.97e-08]]]]
    >>> g_tf_mat = _setup_and_test(lambda x: gkern_tf(2, kernlen = 5,nsigs = x), np.array([1.0]))
    setup_net [(1,)] (5, 5)
    >>> g_np_mat = gkern_nd(2, 5, 1.0)
    >>> pprint(g_tf_mat)
    [[ 0.    0.01  0.02  0.01  0.  ]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.02  0.1   0.16  0.1   0.02]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.    0.01  0.02  0.01  0.  ]]
    >>> pprint(g_np_mat)
    [[ 0.    0.01  0.02  0.01  0.  ]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.02  0.1   0.16  0.1   0.02]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.    0.01  0.02  0.01  0.  ]]
    >>> pprint(np.abs(g_tf_mat - g_np_mat))
    [[  4.27e-11   1.00e-09   4.37e-10   1.00e-09   4.27e-11]
     [  1.00e-09   1.08e-09   3.65e-09   1.08e-09   1.00e-09]
     [  4.37e-10   3.65e-09   3.15e-09   3.65e-09   4.37e-10]
     [  1.00e-09   1.08e-09   3.65e-09   1.08e-09   1.00e-09]
     [  4.27e-11   1.00e-09   4.37e-10   1.00e-09   4.27e-11]]
    """
    pass


def create_dilated_convolutions_weights_2d(in_ch, out_ch, width_x, width_y):
    """
    Create reasonable weights for dilated convolutions so features/structure
    are preserved and neednt be relearned. In the default settings it makes
    the layer return exactly what is passed to it. As it learns this gets
    more complicated
    :param in_ch:
    :param out_ch:
    :param width_x:
    :param width_y:
    :return:
    >>> from keras.models import Sequential
    >>> from keras.layers import Conv2D
    >>> t_model = Sequential()
    >>> cw = create_dilated_convolutions_weights_2d(1, 1, 1, 1)
    >>> tlay = Conv2D(1, kernel_size = (3,3), dilation_rate=(5,5), weights = cw, input_shape = (None, None, 1), padding = 'same')
    >>> t_model.add(tlay)
    >>> out_img = t_model.predict(np.expand_dims(_simple_dist_img,-1))
    >>> np.sum(np.abs(out_img[...,0]-_simple_dist_img))
    0.0
    """
    assert in_ch == out_ch, "In and out should match"
    out_w = np.zeros((2 * width_x + 1, 2 * width_y + 1, in_ch, out_ch),
                     dtype=np.float32)
    out_b = np.zeros(out_ch, dtype=np.float32)
    for i_x, o_x in zip(range(in_ch), range(out_ch)):
        out_w[width_x, width_y, i_x, o_x] = 1.0
    return out_w, out_b


def create_dilated_convolutions_weights_3d(in_ch, out_ch, width_x, width_y,
                                           width_z):
    """
    Create reasonable weights for dilated convolutions so features/structure
    are preserved and neednt be relearned. In the default settings it makes
    the layer return exactly what is passed to it. As it learns this gets
    more complicated
    :param in_ch:
    :param out_ch:
    :param width_x:
    :param width_y:
    :param width_z:
    :return:
    >>> from keras.models import Sequential
    >>> from keras.layers import Conv3D
    >>> t_model = Sequential()
    >>> cw = create_dilated_convolutions_weights_3d(1, 1, 1, 1, 1)
    >>> tlay = Conv3D(1, kernel_size = (3,3,3), dilation_rate=(5,5,5), weights = cw, input_shape = (None, None, None, 1), padding = 'same')
    >>> t_model.add(tlay)
    >>> out_img = t_model.predict(np.expand_dims(np.expand_dims(_simple_dist_img,-1),-1))
    >>> np.sum(np.abs(out_img[...,0,0]-_simple_dist_img))
    0.0
    """
    assert in_ch == out_ch, "In and out should match"
    out_w = np.zeros((2 * width_x + 1, 2 * width_y + 1, 2 * width_z + 1,
                      in_ch, out_ch),
                     dtype=np.float32)
    out_b = np.zeros(out_ch, dtype=np.float32)
    for i_x, o_x in zip(range(in_ch), range(out_ch)):
        out_w[width_x, width_y, width_z, i_x, o_x] = 1.0
    return out_w, out_b


from functools import reduce


def gkern_nd(d=2, kernlen=21, nsigs=3, min_smooth_val=1e-2):
    """
    Create nd gaussian kernels as numpy arrays
    :param d: dimension count
    :param kernlen:
    :param nsigs:
    :param min_smooth_val:
    :return:
    >>> pprint(gkern_nd(3, 3, 1.0))
    [[[ 0.02  0.03  0.02]
      [ 0.03  0.06  0.03]
      [ 0.02  0.03  0.02]]
    <BLANKLINE>
     [[ 0.03  0.06  0.03]
      [ 0.06  0.09  0.06]
      [ 0.03  0.06  0.03]]
    <BLANKLINE>
     [[ 0.02  0.03  0.02]
      [ 0.03  0.06  0.03]
      [ 0.02  0.03  0.02]]]
    >>> pprint(gkern_nd(2, 5, 1.0))
    [[ 0.    0.01  0.02  0.01  0.  ]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.02  0.1   0.16  0.1   0.02]
     [ 0.01  0.06  0.1   0.06  0.01]
     [ 0.    0.01  0.02  0.01  0.  ]]
    >>> pprint(gkern_nd(2, 3, .1))
    [[  3.72e-44   1.93e-22   3.72e-44]
     [  1.93e-22   1.00e+00   1.93e-22]
     [  3.72e-44   1.93e-22   3.72e-44]]
    """
    if type(nsigs) is list:
        assert len(
            nsigs) == d, "Input sigma must be same shape as dimensions {}!={}".format(
            nsigs, d)
    else:
        nsigs = [nsigs] * d
    k_wid = (kernlen - 1) / 2
    all_axs = [np.linspace(-k_wid, k_wid, kernlen)] * d
    all_xxs = np.meshgrid(*all_axs)
    all_dist = reduce(np.add, [
        np.square(cur_xx) / (2 * np.square(np.clip(nsig, min_smooth_val,
                                                   kernlen)))
        for cur_xx, nsig in zip(all_xxs, nsigs)])
    kernel_raw = np.exp(-all_dist)
    return kernel_raw / kernel_raw.sum()


def gkern_tf(d=2, kernlen=21, nsigs=3, norm=True):
    # type: (...) -> tf.Tensor
    """
    Create n-d gaussian kernels as tensors
    :param d: dimension of the kernel
    :param kernlen: length of the kernel (in x, y, z, ...)
    :param nsigs: the sigma values for the kernel either 1 or same length as d
    :return:
    >>> gkern_tf(3, nsigs = tf.placeholder(dtype = tf.float32, shape = (1,)))
    <tf.Tensor 'gaussian_kernel/truediv_3:0' shape=(21, 21, 21) dtype=float32>
    >>> s_range = np.array([1.0])
    >>> _setup_and_test(lambda x: gkern_tf(3, kernlen = 3,nsigs = x), s_range, round = True)
    setup_net [(1,)] (3, 3, 3)
    array([[[ 0.02,  0.03,  0.02],
            [ 0.03,  0.05,  0.03],
            [ 0.02,  0.03,  0.02]],
    <BLANKLINE>
           [[ 0.03,  0.05,  0.03],
            [ 0.05,  0.09,  0.05],
            [ 0.03,  0.05,  0.03]],
    <BLANKLINE>
           [[ 0.02,  0.03,  0.02],
            [ 0.03,  0.05,  0.03],
            [ 0.02,  0.03,  0.02]]])
    >>> _setup_and_test(lambda x: gkern_tf(2, kernlen = 5,nsigs = x), s_range, round = True)
    setup_net [(1,)] (5, 5)
    array([[ 0.  ,  0.01,  0.02,  0.01,  0.  ],
           [ 0.01,  0.05,  0.09,  0.05,  0.01],
           [ 0.02,  0.09,  0.16,  0.09,  0.02],
           [ 0.01,  0.05,  0.09,  0.05,  0.01],
           [ 0.  ,  0.01,  0.02,  0.01,  0.  ]])
    """
    with tf.variable_scope('gaussian_kernel'):
        if type(nsigs) is list:
            assert len(
                nsigs) == d, "Input sigma must be same shape as dimensions {}!={}".format(
                nsigs, d)
        else:
            nsigs = [nsigs] * d
        k_wid = (kernlen - 1) / 2
        all_axs = [tf.linspace(-k_wid, k_wid, kernlen)] * d
        all_xxs = tf.meshgrid(*all_axs, indexing='ij')
        all_dist = reduce(tf.add, [tf.square(cur_xx) / (2 * np.square(nsig))
                                   for cur_xx, nsig in zip(all_xxs, nsigs)])
        kernel_raw = tf.exp(-all_dist)
        if norm:
            kernel_raw = kernel_raw / tf.reduce_sum(kernel_raw)
        return kernel_raw


def label_tf(inp, channel=0, **label_args):
    """
    Connected component labeling as a tensorflow op maps from a black and
    white image to a image with integer specifying components
    :param channel:
    :param label_args:
    :return:
    >>> label_tf(tf.placeholder(dtype = tf.float32, shape = (1, 2, 3, 1)))
    <tf.Tensor 'label_2d/scipy_label:0' shape=(1, 2, 3, 1) dtype=int32>
    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(3),0),-1)
    >>> _setup_and_test(label_tf, s_eye, round = True).squeeze()
    setup_net [(1, 3, 3, 1)] (1, 3, 3, 1)
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])
    >>> s_eye[0,1,1,0] = 0
    >>> _setup_and_test(label_tf, s_eye, round = True).squeeze()
    setup_net [(1, 3, 3, 1)] (1, 3, 3, 1)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  2.]])
    >>> s_eye2 = np.expand_dims(np.expand_dims(np.eye(3)[::-1],0),-1)
    >>> s_full = np.concatenate([s_eye, s_eye2],0) # check two different batch
    >>> _setup_and_test(label_tf, s_full, round = True).squeeze()
    setup_net [(2, 3, 3, 1)] (2, 3, 3, 1)
    array([[[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  2.]],
    <BLANKLINE>
           [[ 0.,  0.,  1.],
            [ 0.,  2.,  0.],
            [ 3.,  0.,  0.]]])
    """
    from scipy.ndimage import label
    def batch_label(x):
        return np.expand_dims(
            np.stack([label(cx[..., channel], **label_args)[0]
                      for cx in x], 0), -1)

    with tf.name_scope('label_2d'):
        y = tf.py_func(batch_label, [inp], tf.int32, name='scipy_label')
        new_shape = inp.get_shape()
        y.set_shape([new_shape[0], new_shape[1], new_shape[2], 1])
        return y


def batch_label_time(in_batch, channel, time_steps, channel_thresh=0.5,
                     **label_args):
    # type: (np.ndarray) -> np.ndarray
    """
    Takes an input and transforms the results into a time series of
    connected component labels
    :param in_batch:
    :param channel:
    :param time_steps:
    :param label_args:
    :return:
    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(3),0),-1)
    >>> pprint(batch_label_time(s_eye, 0, 3).squeeze())
    [[[ 1.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  0.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  1.  0.]
      [ 0.  0.  0.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  1.]]]
    >>> s_eye2 = np.expand_dims(np.expand_dims(np.eye(3)[::-1],0),-1)
    >>> pprint(batch_label_time(np.concatenate([s_eye, s_eye2],0), 0, 3).squeeze())
    [[[[ 1.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  1.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  1.]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[ 0.  0.  1.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  1.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 1.  0.  0.]]]]
    """
    from scipy.ndimage import label
    assert len(in_batch.shape) == 4, "Expected 4D input"
    batch_size, x_wid, y_wid, channels = in_batch.shape
    out_batch = np.zeros((batch_size, time_steps, x_wid, y_wid, 1),
                         dtype=np.float32)
    for i, c_img in enumerate(in_batch):
        c_label = label(c_img[..., channel] > channel_thresh, **label_args)[0]
        for j in range(time_steps):
            out_batch[i, j, :, :, 0] = (
            c_label == (j + 1))  # don't include j=0
    return out_batch


def label_2d_to_time_tf(inp, channel=0, time_steps=5, **label_args):
    """
    Takes an input image, calculates the connected component labels and then
    returns each component as a time-step
    :param inp: image (batch_size, x_wid, y_wid, channels)
    :param channel: channel to use for CCL
    :param time_steps: number of labels to process
    :param time_steps: number of time steps
    :return:
    >>> x = tf.placeholder(dtype = tf.float32, shape = (1, 2, 3, 1))
    >>> label_2d_to_time_tf(x, 0, 3)
    <tf.Tensor 'label_2d_to_time/scipy_batch_label:0' shape=(1, 3, 2, 3, 1) dtype=float32>
    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(3),0),-1)
    >>> _setup_and_test(label_2d_to_time_tf, s_eye, round = True).squeeze()
    setup_net [(1, 3, 3, 1)] (1, 5, 3, 3, 1)
    array([[[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]])
    """
    with tf.name_scope('label_2d_to_time'):
        y = tf.py_func(lambda x: batch_label_time(x, channel=channel,
                                                  time_steps=time_steps,
                                                  **label_args),
                       [inp],
                       tf.float32,
                       name='scipy_batch_label')
        new_shape = inp.get_shape()
        y.set_shape([new_shape[0], time_steps, new_shape[1], new_shape[2], 1])
        return y


def batch_label_time_zoom(in_batch,
                          channel,
                          time_steps,
                          x_size,
                          y_size,
                          channel_thresh=0.5,
                          zoom_order=3,
                          **label_args):
    """

    :param in_batch:
    :param channel:
    :param time_steps:
    :param x_size:
    :param y_size:
    :param channel_thresh:
    :param label_args:
    :return:
    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(2),0),-1)
    >>> s_eye2 = np.roll(s_eye, -1)
    >>> from scipy.ndimage import zoom
    >>> s_full = zoom(np.concatenate([s_eye, s_eye2],-1), [1, 2, 2, 1], order=0)
    >>> pprint(s_full[0, :, :, 0])
    [[ 1.  1.  0.  0.]
     [ 1.  1.  0.  0.]
     [ 0.  0.  1.  1.]
     [ 0.  0.  1.  1.]]
    >>> pprint(s_full[0, :, :, 1])
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    >>> o_mat = batch_label_time_zoom(s_full, 0, 3, 3, 3)
    >>> o_mat.shape
    (1, 3, 3, 3, 2)
    >>> pprint(o_mat[...,0].squeeze())
    [[[ 1.  1.  1.]
      [ 1.  1.  1.]
      [ 1.  1.  1.]]
    <BLANKLINE>
     [[ 1.  1.  1.]
      [ 1.  1.  1.]
      [ 1.  1.  1.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  0.]]]
    >>> pprint(o_mat[...,1].squeeze())
    [[[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  0.]]
    <BLANKLINE>
     [[ 1.  1.  1.]
      [ 1.  1.  1.]
      [ 1.  1.  1.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  0.]]]
    """
    from scipy.ndimage import label, zoom

    assert len(in_batch.shape) == 4, "Expected 4D input"
    batch_size, _, _, channels = in_batch.shape
    out_batch = np.zeros((batch_size, time_steps, x_size, y_size, channels),
                         dtype=np.float32)
    for i, c_img in enumerate(in_batch):
        c_label = label(c_img[..., channel] > channel_thresh, **label_args)[0]
        for j in range(time_steps):
            # don't include j=0
            c_bbox = get_bbox(c_label == (j + 1))
            for ch in range(channels):
                c_ch_roi = apply_bbox(c_img[..., ch], c_bbox)
                if np.prod(c_ch_roi.shape) > 0:
                    zoom_fact = [x_size / c_ch_roi.shape[0],
                                 y_size / c_ch_roi.shape[1]]
                    try:
                        zoom(c_ch_roi,
                             zoom=zoom_fact,
                             output=out_batch[i, j, :, :, ch],
                             order=zoom_order)
                    except Exception as e:
                        warn("""Message: {} cannot be applied to {} because 
                        {}""".format(zoom_fact, c_ch_roi.shape, e),
                             RuntimeWarning)

    return out_batch


def label_time_zoom_tf(inp_batch,
                       channel,
                       time_steps,
                       x_size,
                       y_size,
                       channel_thresh=0.5,
                       **lt_args):
    # type: (...) -> tf.Tensor
    """
    Takes an input image, calculates the connected component labels and then
    returns each component as a time-step
    :param inp_batch: image (batch_size, x_wid, y_wid, channels)
    :param channel:
    :param time_steps:
    :param x_size:
    :param y_size:
    :param channel_thresh:
    :param lt_args:
    :return:
    >>> x = tf.placeholder(dtype = tf.float32, shape = (1, 2, 3, 1))
    >>> label_time_zoom_tf(x, 0, 3, 9, 9)
    <tf.Tensor 'label_zoom_time/scipy_batch_label_zoom:0' shape=(1, 3, 9, 9, 1) dtype=float32>
    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(2),0),-1)
    >>> s_eye2 = np.roll(s_eye, -1)
    >>> from scipy.ndimage import zoom
    >>> s_full = zoom(np.concatenate([s_eye, s_eye2],-1), [1, 2, 2, 1], order=0)
    >>> f = lambda x: label_time_zoom_tf(x, 0, 3, 2, 2)
    >>> _setup_and_test(f, s_full, round = True).squeeze()
    setup_net [(1, 4, 4, 2)] (1, 3, 2, 2, 2)
    array([[[[ 1.,  0.],
             [ 1.,  0.]],
    <BLANKLINE>
            [[ 1.,  0.],
             [ 1.,  0.]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[ 1.,  1.],
             [ 1.,  1.]],
    <BLANKLINE>
            [[ 1.,  1.],
             [ 1.,  1.]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[ 0.,  0.],
             [ 0.,  0.]],
    <BLANKLINE>
            [[ 0.,  0.],
             [ 0.,  0.]]]])
    """
    with tf.name_scope('label_zoom_time'):
        y = tf.py_func(lambda x: batch_label_time_zoom(in_batch=x,
                                                       channel=channel,
                                                       time_steps=time_steps,
                                                       x_size=x_size,
                                                       y_size=y_size,
                                                       channel_thresh=
                                                       channel_thresh,
                                                       **lt_args),
                       [inp_batch],
                       tf.float32,
                       name='scipy_batch_label_zoom')
        new_shape = inp_batch.get_shape()
        y.set_shape([new_shape[0], time_steps, x_size, y_size, new_shape[-1]])
        return y
