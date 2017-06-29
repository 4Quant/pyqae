import itertools as itt

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Convolution2D, BatchNormalization
from keras.layers import Input, Conv3D, ZeroPadding3D, Cropping3D, add, \
    multiply
from keras.layers import Lambda
from keras.models import Model
from keras.models import Sequential

from pyqae.dnn.layers import add_com_phi_grid_tf
from pyqae.dnn.layers import gkern_nd
from pyqae.dnn.layers import gkern_tf
from pyqae.utils import pprint  # noinspection PyUnresolvedReferences

__doc__ = """
A set of neural networks used to generate relevant features for further
analysis. The outputs of the models are meant to be fed into different
models as they provide a better (fully-differentiable) representation which
can be easier for learning and better incorporate relevant spatial features
"""


def mask_net_3d(ishape,
                # type: Tuple[Optional[int], Optional[int], Optional[int], int]
                fg_filt_wid,  # type: Tuple[int, int, int]
                bg_filt_wid,  # type: Tuple[int, int, int]
                trainable=False):
    # type: (...) -> keras.models.Model
    """
    Mask net takes a mask and turns it into a distance map like object using
    uniform filters on the mask and its inverse to represent the
    outside as below [-1, 0) and the
    inside as [0, 1]
    :param ishape:
    :param fg_filt_wid:
    :param bg_filt_wid:
    :param trainable: Should the network be trained with the rest of the model
    :return:
    >>> inet = mask_net_3d((5, 9, 10, 1), (3, 3, 3), (2, 9, 9))
    >>> inet.summary() #doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    RawMask (InputLayer)             (None, 5, 9, 10, 1)   0
    ____________________________________________________________________________________________________
    ExpandingImage_3_9_9 (ZeroPaddin (None, 11, 27, 28, 1) 0
    ____________________________________________________________________________________________________
    InvertedMask (Lambda)            (None, 11, 27, 28, 1) 0
    ____________________________________________________________________________________________________
    BlurMask_7_7_7 (Conv3D)          (None, 11, 27, 28, 1) 343
    ____________________________________________________________________________________________________
    BlurInvMask_5_19_19 (Conv3D)     (None, 11, 27, 28, 1) 1805
    ____________________________________________________________________________________________________
    MaskPositive (Multiply)          (None, 11, 27, 28, 1) 0
    ____________________________________________________________________________________________________
    MaskNegative (Multiply)          (None, 11, 27, 28, 1) 0
    ____________________________________________________________________________________________________
    CombiningImage (Add)             (None, 11, 27, 28, 1) 0
    ____________________________________________________________________________________________________
    CroppingEdges_3_9_9 (Cropping3D) (None, 5, 9, 10, 1)   0
    ====================================================================================================
    Total params: 2,148.0
    Trainable params: 0.0
    Non-trainable params: 2,148.0
    ____________________________________________________________________________________________________
    >>> inet2 = mask_net_3d((None, None, None, 1), (1, 1, 1), (2, 2, 2))
    >>> (100*inet2.predict(np.ones((1, 3, 3, 3, 1))).ravel()).astype(int)
    array([ 29,  44,  29,  44,  66,  44,  29,  44,  29,  44,  66,  44,  66,
           100,  66,  44,  66,  44,  29,  44,  29,  44,  66,  44,  29,  44,  29])

    """
    zp_wid = [max(a, b) for a, b in zip(fg_filt_wid, bg_filt_wid)]
    in_np_mask = Input(shape=ishape, name='RawMask')

    in_mask = ZeroPadding3D(padding=zp_wid,
                            name='ExpandingImage_{}_{}_{}'.format(*zp_wid))(
        in_np_mask)
    inv_mask = Lambda(lambda x: 1.0 - x, name='InvertedMask')(in_mask)
    fg_kernel = np.ones(
        (fg_filt_wid[0] * 2 + 1, fg_filt_wid[1] * 2 + 1, fg_filt_wid[
            2] * 2 + 1))
    fg_kernel = fg_kernel / fg_kernel.sum()
    fg_kernel = np.expand_dims(np.expand_dims(fg_kernel, -1), -1)

    bg_kernel = np.ones((bg_filt_wid[0] * 2 + 1, bg_filt_wid[1] * 2 + 1,
                         bg_filt_wid[2] * 2 + 1))
    bg_kernel = bg_kernel / bg_kernel.sum()
    bg_kernel = np.expand_dims(np.expand_dims(bg_kernel, -1), -1)

    blur_func = lambda name, c_weights: Conv3D(c_weights.shape[-1],
                                               kernel_size=c_weights.shape[:3],
                                               padding='same',
                                               name=name,
                                               activation='linear',
                                               weights=[c_weights],
                                               use_bias=False)

    gmask_in = blur_func('BlurMask_{}_{}_{}'.format(*fg_kernel.shape),
                         fg_kernel)(in_mask)
    gmask_inv = blur_func('BlurInvMask_{}_{}_{}'.format(*bg_kernel.shape),
                          -1 * bg_kernel)(inv_mask)
    gmask_in = multiply([gmask_in, in_mask], name='MaskPositive')
    gmask_inv = multiply([gmask_inv, inv_mask], name='MaskNegative')
    full_img = add([gmask_in, gmask_inv], name='CombiningImage')
    full_img = Cropping3D(cropping=zp_wid,
                          name='CroppingEdges_{}_{}_{}'.format(*zp_wid))(
        full_img)
    out_model = Model(inputs=[in_np_mask], outputs=[full_img])
    out_model.trainable = trainable
    for ilay in out_model.layers:
        ilay.trainable = trainable
    return out_model


from keras.layers import GlobalAveragePooling2D


def GlobalSpreadAverage2D(suffix):
    """
    A global average pooling that is then spread as the same value across
    all pixels
    :param suffix:
    :return:
    >>> from keras.models import Sequential
    >>> in_node = Input(shape = (4, 4, 3))
    >>> n_node = GlobalSpreadAverage2D(suffix='_GSA')(in_node)
    >>> t_model = Model(inputs = [in_node], outputs = [n_node])
    >>> t_model.summary() # doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 4, 4, 3)           0
    _________________________________________________________________
    GAP_GSA (GlobalAveragePoolin (None, 3)                 0
    _________________________________________________________________
    DeepMiddle_GSA (Lambda)      (None, 4, 4, 6)           0
    =================================================================
    Total params: 0.0
    Trainable params: 0.0
    Non-trainable params: 0.0
    _________________________________________________________________
    """

    def _deep_middle(in_block):
        in_gap, in_tensor = in_block
        in_gap = tf.expand_dims(tf.expand_dims(in_gap, 1), 1)
        ga_wid = tf.shape(in_gap)[3]
        x_wid, y_wid = tf.shape(in_tensor)[1], tf.shape(in_tensor)[2]
        tile_mid = tf.tile(in_gap, (1, x_wid, y_wid, 1))
        return tf.concat([in_tensor, tile_mid], -1)

    def _layer_code(in_tensor):
        ga_layer = GlobalAveragePooling2D(name="GAP{}".format(suffix))(
            in_tensor)
        return Lambda(_deep_middle, name='DeepMiddle{}'.format(suffix))(
            [ga_layer, in_tensor])

    return _layer_code


def PhiComGrid3DLayer(z_rad, include_r, **args):
    """
    A PhiComGrid layer based on the add_com_phi_grid_tf function which take
    the center of mass of the object in 3D and creates a arcsin map for the
    3 coordinates
    :param z_rad: the radius near the middle to black out (force 0)
    :param args:
    :return:
    >>> from keras.models import Sequential
    >>> t_model = Sequential()
    >>> t_model.add(PhiComGrid3DLayer(z_rad=0.0, input_shape=(None, None, None, 1), name='PhiGrid', include_r = False))
    >>> t_model.summary() # doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    PhiGrid (Lambda)             (None, None, None, None,  0
    =================================================================
    Total params: 0.0
    Trainable params: 0.0
    Non-trainable params: 0.0
    _________________________________________________________________
    >>> out_res = t_model.predict(np.ones((1, 3, 3, 3, 1)))
    >>> out_res.shape
    (1, 3, 3, 3, 3)
    >>> pprint(out_res[0,:,0,0,0])
    [-0.2  0.   0.2]
    >>> pprint(out_res[0,:,0,0,1])
    [-0.2  -0.25 -0.2 ]
    """
    return Lambda(lambda x: add_com_phi_grid_tf(x,
                                                z_rad=z_rad,
                                                include_r=include_r),
                  **args)


def get_diff_vecs(vec_count, loop=True):
    c_var = True
    while c_var:
        for base_img, diff_img in itt.combinations(range(vec_count), 2):
            c_vec = np.zeros((vec_count, 1))
            c_vec[base_img] = 1
            c_vec[diff_img] = -1
            yield c_vec
        if not loop:
            c_var = False


def get_diff_mat(vec_count, diff_count, loop=True):
    return np.hstack([c_vec for c_vec, _ in
                      zip(get_diff_vecs(vec_count, loop=loop),
                          range(diff_count))])


def dog_net_2d(gk_count,
               dk_count,
               min_width,
               max_width,
               d_noise=0,
               input_shape=(None, None, 1),
               static_diff_mat=False,
               add_bn_layer=True,
               train_filters=False,
               train_differences=True,
               k_dim=None):
    # type: (...) -> keras.models.Model
    """
    Create a differentiable Difference of Gaussians network for trainable
    spot-detection
    :param gk_count: number of filters
    :param dk_count: number of difference steps
    :param min_width: minimum filter width
    :param max_width: maximum filter width
    :param d_noise: noise to add to the difference layer
    :param input_shape:
    :param static_diff_mat:
    :param add_bn_layer:
    :param train_filters: train the convolutional kernels
    :param train_differences: train the difference matrix
    :param k_dim: manually specify the kernel dimension
    :return:
    >>> dog_model = dog_net_2d(2, 1, 0.5, 2)
    >>> dog_model.summary() #doctest: +NORMALIZE_WHITESPACE
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    AllGaussianStep (Conv2D)     (None, None, None, 2)     18
    _________________________________________________________________
    DifferenceStep (Conv2D)      (None, None, None, 1)     2
    _________________________________________________________________
    NormalizeDiffModel (BatchNor (None, None, None, 1)     4
    =================================================================
    Total params: 24.0
    Trainable params: 4.0
    Non-trainable params: 20.0
    _________________________________________________________________
    >>> in_img = 1000*np.expand_dims(np.expand_dims(np.eye(3), 0),-1)
    >>> iv = dog_model.predict(in_img)
    >>> iv[0,:,:,0].astype(int) #doctest: +NORMALIZE_WHITESPACE
    array([[397, -63, -90],
           [-63, 307, -63],
           [-90, -63, 397]])
    """
    if k_dim is None:
        k_dim = int(np.clip((max_width - 2) * 2 + 1, 3, 9e9))
    c_weights = np.expand_dims(
        np.stack([gkern_nd(d=2, kernlen=k_dim, nsigs=nsig) for nsig in
                  np.linspace(min_width, max_width, gk_count)], -1),
        -2)
    if static_diff_mat:
        orig_pad = np.eye(gk_count)[:, 1:]
        diff_pad = np.zeros((gk_count, gk_count - 1))
        np.fill_diagonal(diff_pad, -1)
        d_weights = np.expand_dims(np.expand_dims(orig_pad + diff_pad, 0), 0)
    else:
        d_weights = np.expand_dims(
            np.expand_dims(get_diff_mat(gk_count, dk_count, loop=False), 0), 0)
    if d_noise > 0:
        d_weights = d_weights + np.random.uniform(-d_noise, d_noise,
                                                  size=d_weights.shape)
    is_layer = dict(input_shape=input_shape)
    pet_ddog_net = Sequential()
    pet_ddog_net.add(Convolution2D(gk_count,
                                   kernel_size=(k_dim, k_dim),
                                   weights=[c_weights],
                                   use_bias=False,
                                   name='AllGaussianStep',
                                   padding='same',
                                   activation='linear', **is_layer))
    pet_ddog_net.add(Convolution2D(d_weights.shape[-1],
                                   kernel_size=(1, 1),
                                   weights=[d_weights],
                                   use_bias=False,
                                   name='DifferenceStep',
                                   padding='same',
                                   activation='linear'))
    if add_bn_layer:
        pet_ddog_net.add(BatchNormalization(name='NormalizeDiffModel'))
    pet_ddog_net.layers[0].trainable = train_filters
    pet_ddog_net.layers[1].trainable = train_differences
    return pet_ddog_net


def vdog_net_2d(gk_count,
                dk_count,
                min_width,
                max_width,
                d_noise=0,
                input_shape=(None, None, 1),
                # type: Tuple[Optional[int], Optional[int], int]
                static_diff_mat=False,
                add_bn_layer=True,
                k_dim=None,
                train_differences=True,
                suffix=''):
    # type: (...) -> keras.models.Model
    """
    Create a differentiable Difference of Gaussians network that scales
    with voxel size for trainable, resolution independent spot-detection
    :param gk_count: number of filters
    :param dk_count: number of difference steps
    :param min_width: minimum filter width
    :param max_width: maximum filter width
    :param d_noise: noise to add to the difference layer
    :param input_shape:
    :param static_diff_mat:
    :param add_bn_layer:
    :param train_filters: train the convolutional kernels
    :param train_differences: train the difference matrix
    :param k_dim: manually specify the kernel dimension
    :return:
    >>> dog_model = vdog_net_2d(2, 1, 1.0, 2.0, k_dim = 21, add_bn_layer = False)
    >>> dog_model.summary() #doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    InputImage (InputLayer)          (None, None, None, 1) 0
    ____________________________________________________________________________________________________
    XY_VoxDims (InputLayer)          (None, 2)             0
    ____________________________________________________________________________________________________
    AllGaussianStep (Lambda)         (None, None, None, 2) 0
    ____________________________________________________________________________________________________
    DifferenceStep (Conv2D)          (None, None, None, 1) 2
    ====================================================================================================
    Total params: 2.0
    Trainable params: 2.0
    Non-trainable params: 0.0
    ____________________________________________________________________________________________________
    >>> in_img = 1*np.expand_dims(np.expand_dims(np.eye(15), 0),-1)
    >>> in_vox = np.array([1.0, 1.0]).reshape((1, 2))
    >>> iv = dog_model.predict([in_img, in_vox])
    >>> iv.shape
    (1, 15, 15, 1)
    >>> pprint(iv[0,5:-5,5:-5,0]) #doctest: +NORMALIZE_WHITESPACE
    [[ 0.14  0.09 -0.01 -0.05 -0.05]
     [ 0.09  0.14  0.09 -0.01 -0.05]
     [-0.01  0.09  0.14  0.09 -0.01]
     [-0.05 -0.01  0.09  0.14  0.09]
     [-0.05 -0.05 -0.01  0.09  0.14]]
    >>> from scipy.ndimage import zoom # show the results are resolution indepndent
    >>> in_vox2 = np.array([0.5, 0.5]).reshape((1, 2))
    >>> in_img2 = zoom(in_img, (1, 2, 2, 1), order = 2)
    >>> iv2 = dog_model.predict([in_img2, in_vox2])
    >>> iv2z = zoom(iv2, (1, 0.5, 0.5, 1), order = 2)
    >>> pprint(iv2z[0,5:-5,5:-5,0]) #doctest: +NORMALIZE_WHITESPACE
    [[ 0.14  0.08 -0.01 -0.05 -0.05]
     [ 0.08  0.14  0.08 -0.01 -0.05]
     [-0.01  0.08  0.14  0.08 -0.01]
     [-0.05 -0.01  0.08  0.14  0.08]
     [-0.05 -0.05 -0.01  0.08  0.14]]
    >>> pprint(np.mean(np.abs(iv2z/iv)[:,5:-5, 5:-5], (0, 3)))
    [[ 1.    0.97  1.98  1.07  0.98]
     [ 0.97  1.    0.97  1.98  1.07]
     [ 1.98  0.97  1.    0.97  1.98]
     [ 1.07  1.98  0.97  1.    0.97]
     [ 0.98  1.07  1.98  0.97  1.  ]]
    >>> pprint(np.mean(np.abs(iv2z-iv)[:,5:-5, 5:-5], (0, 3)))
    [[ 0.    0.    0.01  0.    0.  ]
     [ 0.    0.    0.    0.01  0.  ]
     [ 0.01  0.    0.    0.    0.01]
     [ 0.    0.01  0.    0.    0.  ]
     [ 0.    0.    0.01  0.    0.  ]]
    """
    if k_dim is None:
        k_dim = int(np.clip((max_width - 2) * 2 + 1, 3, 9e9))

    raw_img = Input(shape=input_shape, name='InputImage{}'.format(suffix))
    vox_size = Input(shape=(2,), name='XY_VoxDims{}'.format(suffix))
    if static_diff_mat:
        orig_pad = np.eye(gk_count)[:, 1:]
        diff_pad = np.zeros((gk_count, gk_count - 1))
        np.fill_diagonal(diff_pad, -1)
        d_weights = np.expand_dims(np.expand_dims(orig_pad + diff_pad, 0), 0)
    else:
        d_weights = np.expand_dims(
            np.expand_dims(get_diff_mat(gk_count, dk_count, loop=False), 0), 0)
    if d_noise > 0:
        d_weights = d_weights + np.random.uniform(-d_noise, d_noise,
                                                  size=d_weights.shape)

    def var_rad_layer(x):
        in_img, in_vox = x

        tf_weights = tf.expand_dims(
            tf.stack([gkern_tf(d=2,
                               kernlen=k_dim,
                               nsigs=[c_wid / vox_size[0, 0],
                                      c_wid / vox_size[0, 1]])
                      for c_wid in
                      np.linspace(min_width, max_width, gk_count)], -1),
            -2)
        return K.conv2d(in_img, kernel=tf_weights, padding='same')

    gauss_layer = Lambda(var_rad_layer,
                         name='AllGaussianStep{}'.format(suffix))(
        [raw_img, vox_size])
    diff_layer = Convolution2D(d_weights.shape[-1],
                               kernel_size=(1, 1),
                               weights=[d_weights],
                               use_bias=False,
                               name='DifferenceStep{}'.format(suffix),
                               padding='same',
                               activation='linear')(gauss_layer)

    if add_bn_layer:
        diff_layer = BatchNormalization(
            name='NormalizeDiffModel{}'.format(suffix))(diff_layer)

    out_model = Model(inputs=[raw_img, vox_size], outputs=[diff_layer])
    out_model.layers[1].trainable = train_differences
    return out_model


def vdog_net_3d(gk_count,  # type: int
                dk_count,  # type: int
                min_width,  # type: float
                max_width,  # type: float
                d_noise=0,  # type: float
                input_shape=(None, None, None, 1),
                # type: Tuple[Optional[int], Optional[int], Optional[int], int]
                static_diff_mat=False,
                add_bn_layer=True,
                k_dim=None,  # type: Optional[int]
                train_differences=True,
                suffix=''):
    # type: (...) -> keras.models.Model
    """
    Create a differentiable Difference of Gaussians network that scales
    with voxel size for trainable, resolution independent spot-detection
    :param gk_count: number of filters
    :param dk_count: number of difference steps
    :param min_width: minimum filter width
    :param max_width: maximum filter width
    :param d_noise: noise to add to the difference layer
    :param input_shape:
    :param static_diff_mat:
    :param add_bn_layer:
    :param train_filters: train the convolutional kernels
    :param train_differences: train the difference matrix
    :param k_dim: manually specify the kernel dimension
    :return:
    >>> dog_model = vdog_net_3d(2, 1, 1.0, 2.0, k_dim = 21, add_bn_layer = False)
    >>> dog_model.summary() #doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    InputImage (InputLayer)          (None, None, None, No 0
    ____________________________________________________________________________________________________
    XYZ_VoxDims (InputLayer)         (None, 3)             0
    ____________________________________________________________________________________________________
    AllGaussianStep (Lambda)         (None, None, None, No 0
    ____________________________________________________________________________________________________
    DifferenceStep (Conv3D)          (None, None, None, No 2
    ====================================================================================================
    Total params: 2.0
    Trainable params: 2.0
    Non-trainable params: 0.0
    ____________________________________________________________________________________________________
    >>> in_img = np.expand_dims(np.expand_dims(np.expand_dims(np.eye(15), 0),-1), -1)
    >>> in_vox = np.array([1.0, 1.0, 1.0]).reshape((1, 3))
    >>> iv = dog_model.predict([in_img, in_vox])
    >>> iv.shape
    (1, 15, 15, 1, 1)
    >>> pprint(iv[0,5:-5,5:-5, 0 ,0]) #doctest: +NORMALIZE_WHITESPACE
    [[ 0.08  0.06  0.02 -0.   -0.01]
     [ 0.06  0.08  0.06  0.02 -0.  ]
     [ 0.02  0.06  0.08  0.06  0.02]
     [-0.    0.02  0.06  0.08  0.06]
     [-0.01 -0.    0.02  0.06  0.08]]
    >>> from scipy.ndimage import zoom # show the results are resolution indepndent
    >>> in_vox2 = np.array([0.5, 0.5, 1]).reshape((1, 3))
    >>> in_img2 = zoom(in_img, (1, 2, 2, 1, 1), order = 2)
    >>> iv2 = dog_model.predict([in_img2, in_vox2])
    >>> iv2z = zoom(iv2, (1, 0.5, 0.5, 1, 1), order = 2)
    >>> pprint(iv2z[0,5:-5,5:-5,0,0]) #doctest: +NORMALIZE_WHITESPACE
    [[ 0.09  0.06  0.02 -0.01 -0.01]
     [ 0.06  0.09  0.06  0.02 -0.01]
     [ 0.02  0.06  0.09  0.06  0.02]
     [-0.01  0.02  0.06  0.09  0.06]
     [-0.01 -0.01  0.02  0.06  0.09]]
    >>> pprint(np.mean(np.abs(iv2z/iv)[:,5:-5, 5:-5, 0], (0, 3)))
    [[ 1.02  1.    0.91  1.33  1.01]
     [ 1.    1.02  1.    0.91  1.33]
     [ 0.91  1.    1.02  1.    0.91]
     [ 1.33  0.91  1.    1.02  1.  ]
     [ 1.01  1.33  0.91  1.    1.02]]
    >>> pprint(np.mean(np.abs(iv2z-iv)[:,5:-5, 5:-5, 0], (0, 3)))
    [[ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    """
    if k_dim is None:
        k_dim = int(np.clip((max_width - 2) * 2 + 1, 3, 9e9))

    raw_img = Input(shape=input_shape, name='InputImage{}'.format(suffix))
    vox_size = Input(shape=(3,), name='XYZ_VoxDims{}'.format(suffix))
    if static_diff_mat:
        orig_pad = np.eye(gk_count)[:, 1:]
        diff_pad = np.zeros((gk_count, gk_count - 1))
        np.fill_diagonal(diff_pad, -1)
        d_weights = np.expand_dims(np.expand_dims(orig_pad + diff_pad, 0), 0)
    else:
        d_weights = np.expand_dims(
            np.expand_dims(
                np.expand_dims(get_diff_mat(gk_count, dk_count, loop=False),
                               0),
                0), 0)
    if d_noise > 0:
        d_weights = d_weights + np.random.uniform(-d_noise, d_noise,
                                                  size=d_weights.shape)

    def var_rad_layer(x):
        in_img, in_vox = x

        tf_weights = tf.expand_dims(
            tf.stack([gkern_tf(d=3,
                               kernlen=k_dim,
                               nsigs=[c_wid / vox_size[0, 0],
                                      c_wid / vox_size[0, 1],
                                      c_wid / vox_size[0, 2]])
                      for c_wid in
                      np.linspace(min_width, max_width, gk_count)], -1),
            -2)
        return K.conv3d(in_img, kernel=tf_weights, padding='same')

    gauss_layer = Lambda(var_rad_layer,
                         name='AllGaussianStep{}'.format(suffix))(
        [raw_img, vox_size])
    diff_layer = Conv3D(d_weights.shape[-1],
                        kernel_size=(1, 1, 1),
                        weights=[d_weights],
                        use_bias=False,
                        name='DifferenceStep{}'.format(suffix),
                        padding='same',
                        activation='linear')(gauss_layer)

    if add_bn_layer:
        diff_layer = BatchNormalization(
            name='NormalizeDiffModel{}'.format(suffix))(diff_layer)

    out_model = Model(inputs=[raw_img, vox_size], outputs=[diff_layer])
    out_model.layers[1].trainable = train_differences
    return out_model
