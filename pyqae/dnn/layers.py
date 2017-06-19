import numpy as np
import tensorflow as tf

__doc__ = """
A set of Tensorflow-based layers and operations for including in models
allowing meaningful spatial and medical information to be included in images
"""


def _setup_and_test(in_func, *in_arrs):
    """
    For setting up a simple graph and testing it
    :param in_func:
    :param in_arr:
    :return:
    """
    with tf.Graph().as_default() as g:
        in_vals = [tf.placeholder(dtype=tf.float32, shape=in_arr.shape) for
                   in_arr in in_arrs]
        out_val = in_func(*in_vals)
        print('setup_net', [in_arr.shape for in_arr in in_arrs], out_val.shape)
    with tf.Session(graph=g) as c_sess:
        return c_sess.run(fetches=[out_val],
                          feed_dict={in_val: in_arr
                                     for in_val, in_arr in
                                     zip(in_vals, in_arrs)})[0]


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


def add_com_grid_tf(in_layer,
                    layer_concat=False
                    ):
    # type: (tf.Tensor, bool) -> tf.Tensor
    """
    Adds spatial grids to images for making segmentation easier
    This particular example utilizes the image-weighted center of mass by
    summing the input layer across the channels

    :param in_layer:
    :param layer_concat:
    :return:
    >>> out_img = _setup_and_test(add_com_grid_tf, np.ones((5, 4, 3, 2, 1)))
    setup_net [(5, 4, 3, 2, 1)] (5, ?, ?, ?, 3)
    >>> out_img.shape
    (5, 4, 3, 2, 3)
    >>> out_img[0, :, 0, 0, 0]
    array([-1.        , -0.33333334,  0.33333334,  1.        ], dtype=float32)
    >>> out_img[0, 0, :, 0, 1]
    array([-1.,  0.,  1.], dtype=float32)
    >>> out_img[0, 0, 0, :, 2]
    array([-1.,  1.], dtype=float32)
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
            expand_op = lambda iv: tf.expand_dims(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(iv, -1), -1), -1),
                -1)
            tile_op = lambda iv: tf.tile(iv, [1, xg_wid, yg_wid, zg_wid, 1])
            res_op = lambda iv: tile_op(expand_op(iv))

            sm_matlist = [res_op(c_var) for c_var in sm_list]

        with tf.variable_scope('make_grid'):
            out_var = [tf.reshape(c_var, (
                batch_size, xg_wid, yg_wid, zg_wid, 1)) - c_sm
                       for c_var, c_sm in zip(svar_list, sm_matlist)]

            xy_vec = tf.concat(out_var, -1)
            # txy_vec = tf.tile(xy_vec, [batch_size, 1, 1, 1, 1])

        if layer_concat:
            return tf.concat([in_layer, xy_vec], -1)
        else:
            return xy_vec
