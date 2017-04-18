from __future__ import print_function, division, absolute_import

__doc__ = """
PETNET is the general tool powering the deep learning algorithms for PETCT and other images
The unique aspect of PETCT is the ability to process data in different chains to
 allow for multiple types of information to be easily integrated
"""

import os
import warnings

try:
    if os.environ[
        'KERAS_BACKEND'] == 'theano':
        warnings.warn("Theano backend is expected!", RuntimeWarning)
except KeyError:
    print("Backend for keras is undefined setting to theano for PETNET")
    os.environ['KERAS_BACKEND'] = 'theano'

from keras import backend as K

K.set_image_dim_ordering('th')
from keras.layers import Cropping2D

from keras.models import Model
from keras.layers import Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from collections import defaultdict
import numpy as np
from pyqae.utils import Tuple, List, Dict, Any, Optional
from pyqae.dnn import fix_name_tf


def _build_nd_umodel(in_shape,  # type: Tuple[int, int]
                     layers,  # type: int
                     branches,  # type: List[Tuple[str, int, int]]
                     conv_op, pool_op, upscale_op, crop_op,
                     layer_size_fcn=lambda i: 3,
                     pool_size=2,
                     dropout_rate=0.0,
                     last_layer_depth=4,
                     crop_mode=True,
                     use_b_conv=False,
                     use_b_conv_out=False,  # type: Optional[bool]
                     use_bn=True,
                     verbose=False,
                     single_input=False,
                     use_deconv=False,
                     input_dict=None  # type: Dict[str, Tuple[Object, Object]]
                     ):
    """

    :param in_shape:
    :param layers:
    :param branches:
    :param conv_op:
    :param pool_op:
    :param upscale_op:
    :param crop_op:
    :param layer_size_fcn:
    :param pool_size:
    :param dropout_rate:
    :param last_layer_depth:
    :param crop_mode:
    :param use_b_conv:
    :param use_b_conv_out:
    :param use_bn:
    :param verbose:
    :param single_input:
    :param input_dict: a dictionary of tuples containing the name of an
    input as well as the original input and the node to put into the
    network, it allows preprocessing steps to be done elsewhere
    :return:
    """
    border_mode = 'valid' if crop_mode else 'same'
    use_b_conv_out = use_b_conv_out if use_b_conv_out is not None else use_b_conv
    x_wid, y_wid = in_shape
    branch_dict = defaultdict(dict)  # type: Dict[str, Dict[str, Any]]

    input_list = []
    if input_dict is None:
        input_dict = {}

    def _make_input(*args, name='emptyInput', **kwargs):
        # keep track of all the inputs
        if name in input_dict:
            in_node, cur_node = input_dict[name]
        else:
            cur_node = Input(*args, name=name, **kwargs)
            in_node = cur_node

        input_list.append(in_node)
        return cur_node

    if single_input:
        # this only works with theano ordering
        full_ch_cnt = sum(
            [branch_ch for branch_name, branch_ch, branch_depth in branches])
        full_input = _make_input((full_ch_cnt, x_wid, y_wid))
        in_branches = []
        start_idx = 0
        for branch_name, branch_ch, branch_depth in branches:
            in_branches += [
                Lambda(lambda x: x[:, start_idx:(start_idx + branch_ch), :, :],
                       output_shape=(branch_ch, x_wid, y_wid),
                       name=fix_name_tf(branch_name))(full_input)]
            start_idx += branch_ch
    else:
        in_branches = [_make_input((branch_ch, x_wid, y_wid),
                                   name=fix_name_tf(branch_name))
                       for branch_name, branch_ch, branch_depth in branches]

    for (branch_name, branch_ch, branch_depth), input_im in zip(branches,
                                                                in_branches):
        branch_id = fix_name_tf(branch_name)
        if branch_depth > 0:
            fmap = conv_op(branch_depth, 3,
                           name='A {} Prefiltering'.format(branch_id),
                           border_mode=border_mode)(
                input_im)
            first_layer = conv_op(branch_depth, 3,
                                  name='B {} Prefiltering'.format(branch_id),
                                  border_mode=border_mode)(
                fmap)
        else:
            if crop_mode:
                first_layer = crop_op(lay_kern_wid // 2,
                                      'A {} PreSkip Cropping'.format(
                                          branch_id))(
                    input_im)
                first_layer = crop_op(lay_kern_wid // 2,
                                      'B {} PreSkip Cropping'.format(
                                          branch_id))(first_layer)
            else:
                first_layer = input_im

        last_layer = first_layer
        conv_layers = []
        pool_layers = []

        # save_layer = crop_op(1, 'Cropping')(last_layer) if crop_mode else last_layer
        conv_layers += [last_layer]

        # the image layers
        for ilay in range(layers):

            # only apply BN to the left pathway

            pool_layers += [pool_op(pool_size,
                                    name='{} Downscaling [{}]'.format(
                                        branch_id, ilay))(last_layer)]
            if use_bn:
                last_layer = BatchNormalization(
                    axis=1,  # TODO: change this for tensorflow
                    name=fix_name_tf(
                        '{} Batch Normalization [{}]'.format(branch_id,
                                                             ilay)))(
                    pool_layers[-1])
            else:
                last_layer = pool_layers[-1]
            lay_depth = branch_depth * np.power(2, ilay + 1)

            conv_layers += [last_layer]

            # double filters

            # use a different width for the last layer (needed if it is 1x1 convolution)
            if ilay == (layers - 1):
                lay_kern_wid = layer_size_fcn(ilay)
            else:
                lay_kern_wid = layer_size_fcn(ilay)

            if lay_depth > 0:
                post_conv_step = conv_op(lay_depth, lay_kern_wid,
                                         border_mode=border_mode,
                                         name='A {} Feature Maps [{}]'.format(
                                             branch_id, ilay, lay_depth))(
                    last_layer)

                if use_b_conv: post_conv_step = conv_op(lay_depth,
                                                        lay_kern_wid,
                                                        border_mode=border_mode,
                                                        name='B {} Feature-Maps [{}]'.format(
                                                            branch_id, ilay,
                                                            lay_depth))(
                    post_conv_step)
            else:
                if crop_mode:
                    post_conv_step = crop_op(lay_kern_wid // 2,
                                             'A {} Skip Cropping [{}]'.format(
                                                 branch_id, ilay))(
                        last_layer)
                    if use_b_conv:
                        post_conv_step = crop_op(lay_kern_wid // 2,
                                                 'B {} Skip Cropping [{}]'.format(
                                                     branch_id, ilay))(
                            post_conv_step)
                else:
                    post_conv_step = last_layer

            last_layer = post_conv_step

        # remove the last layer
        if verbose: print('image layers',
                          [i._keras_shape for i in conv_layers])
        rev_layers = list(
            reversed(list(zip(range(layers + 1), conv_layers[:]))))
        # rev_layers += [(-1,first_layer)]
        branch_dict[branch_id]['input'] = input_im
        branch_dict[branch_id]['rev_layers'] = rev_layers

    all_rev_layers = [c_branch['rev_layers'] for c_branch in
                      branch_dict.values()]
    all_inputs = [c_branch['input'] for c_branch in branch_dict.values()]
    last_layer = None
    out_depth = np.sum([depth for _, _, depth in
                        branches])  # just assume it is the sum of input depths

    for cur_rev_layers in zip(*all_rev_layers):

        merge_layers = [] if last_layer is None else [last_layer]
        if False:
            out_layers = [i_pool for
                          (branch_name, _, branch_depth), (ilay, i_pool) in
                          zip(branches, cur_rev_layers)]
            return Model(input=all_inputs, output=out_layers)
        for (branch_name, _, branch_depth), (ilay, i_pool) in zip(branches,
                                                                  cur_rev_layers):
            branch_id = fix_name_tf(branch_name)
            lay_depth = out_depth * np.power(2, ilay)
            if verbose: print(ilay, i_pool._keras_shape)

            if crop_mode and (len(merge_layers) > 0) and (
                        last_layer is not None):
                assert last_layer is not None, "Last layer cannot be empty! {}, #{}, pool:{}".format(
                    merge_layers, ilay,
                    i_pool)
                crop_d = int(np.ceil(
                    (i_pool._keras_shape[2] - last_layer._keras_shape[2]) / 2))
                if crop_d > 0:
                    if verbose: print('Shape Mismatch:',
                                      i_pool._keras_shape[2],
                                      last_layer._keras_shape[2], 'crop->',
                                      crop_d)
                    i_pool = crop_op(crop_d,
                                     '{} Cropping [{}]'.format(branch_id,
                                                               ilay))(i_pool)
            merge_layers += [i_pool]

        if verbose: print('merging:', [x._keras_shape for x in merge_layers])
        if len(merge_layers) > 1:
            cur_merge = concatenate(merge_layers,
                                    axis=1,
                                    name=fix_name_tf(
                                        'Mixing Original Data [{}]'.format(
                                            ilay + 1)))
        else:
            cur_merge = merge_layers[0]

        last_layer = cur_merge

        if ilay > 0:
            cur_up = upscale_op(lay_depth, pool_size, name='Upsampling [{'
                                                           '}]'.format(
                ilay + 1))(last_layer)
            last_layer = cur_up

            if dropout_rate > 0: last_layer = Dropout(dropout_rate,
                                                      name=fix_name_tf(
                                                          'Random Removal [{}]'.format(
                                                              ilay + 1)))(
                last_layer)

            if not use_deconv:
                a_conv = conv_op(lay_depth, 3,
                                 name='A Deconvolution [{}]'.format(ilay + 1),
                                 border_mode=border_mode)(
                    last_layer)

                if use_b_conv_out:
                    b_conv = conv_op(lay_depth, 3,
                                     name='B Deconvolution {}]'.format(
                                         ilay + 1), border_mode=border_mode)(
                        a_conv)
                    last_layer = b_conv
                else:
                    last_layer = a_conv

    out_conv = conv_op(last_layer_depth, 1, activation='sigmoid',
                       name='Calculating Probability',
                       border_mode=border_mode)(last_layer)

    model = Model(inputs=input_list, outputs=out_conv)

    return model


def build_2d_umodel(in_img,
                    layers,
                    branches=[('Chest XRay', 1, 8), ('Mesh', 2, 4)],
                    lsf=lambda i: 3,
                    pool_size=2,
                    dropout_rate=0,
                    last_layer_depth=1,
                    use_bn=True,
                    crop_mode=True,
                    use_b_conv=True,
                    verbose=False,
                    single_input=False,
                    input_dict=None,
                    use_leakyrelu=False,
                    use_deconv=False
                    ):
    """

    :param in_img:
    :param layers:
    :param lsf:
    :param pool_size:
    :param dropout_rate:
    :param last_layer_depth:
    :param branches:
    :param input_dict: a dictionary of input nodes
    :return:
    >>> simple_mod = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest XRay', 1, 2)], crop_mode = False, use_bn = False, verbose = False, use_deconv=False)
    >>> len(simple_mod.layers)
    8
    >>> simple_mod.layers[-1].output_shape[1:]
    (1, 32, 32)
    >>> len(build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest XRay', 1, 2)], crop_mode = False, use_bn = True, verbose = False).layers)
    9
    >>> len(build_2d_umodel(np.zeros((1,1,32,32)), 2, [('Chest XRay', 1, 2)], crop_mode = False, use_bn = True, verbose = False).layers)
    16
    >>> better_model = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest CT', 1, 8), ('PET', 1, 4)], crop_mode = True, use_bn = False, verbose = False)
    >>> better_model.layers[-1].output_shape[1:]
    (1, 26, 26)
    >>> bestest_model = build_2d_umodel(np.zeros((1,1,64,64)), 2, [('Chest CT', 1, 8), ('PET', 1, 4)], crop_mode = True, use_bn = True, verbose = False)
    >>> len(bestest_model.layers)
    30
    >>> bestest_model.layers[-1].output_shape[1:]
    (1, 46, 46)
    >>> fat_model = build_2d_umodel(np.zeros((1,1,128,128)), 3, [('Chest CT', 1, 8), ('PET', 1, 4), ('POS', 3, 3)], crop_mode = False, use_bn = False, verbose = False)
    >>> len(fat_model.layers)
    41
    >>> fat_model.layers[-1].output_shape[1:]
    (1, 128, 128)
    >>> [x.output_shape[1:] for x in fat_model.layers if x.name.find('Mixing')>=0]
    [(60, 16, 16), (150, 32, 32), (75, 64, 64), (45, 128, 128)]
    >>> fat_crop_model = build_2d_umodel(np.zeros((1,1,132,132)), 3, [('Chest CT', 1, 8), ('PET', 1, 4), ('POS', 3, 3)], crop_mode = True, use_bn = False, verbose = False)
    >>> len(fat_crop_model.layers)
    50
    >>> fat_crop_model.layers[-1].output_shape[1:]
    (1, 90, 90)
    >>> [x.output_shape[1:] for x in fat_crop_model.layers if x.name.find('Mixing')>=0]
    [(60, 13, 13), (150, 24, 24), (75, 46, 46), (45, 90, 90)]
    >>> si_model = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest CT', 1, 8), ('PET', 2, 4)], crop_mode = True, use_bn = False, verbose = False, single_input = True)
    >>> si_model.layers[0].output_shape[1:]
    (3, 32, 32)
    >>> si_model.layers[-1].output_shape[1:]
    (1, 26, 26)
    >>> ct_pos = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest CT', 1, 8), ('XYZPos', 3, 0)], crop_mode = False, use_bn = False, verbose = False, single_input = True)
    >>> len(ct_pos.layers)
    12
    >>> ct_pos.layers[0].output_shape[1:]
    (4, 32, 32)
    >>> ct_pos.layers[-1].output_shape[1:]
    (1, 32, 32)
    >>> cct_pos = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest CT', 1, 8), ('XYZPos', 3, 0)], crop_mode = True, use_bn = False, verbose = False, single_input = True)
    >>> len(cct_pos.layers)
    16
    >>> cct_pos.layers[-1].output_shape[1:]
    (1, 26, 26)
    >>> simple_mod = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest XRay', 1, 2)], crop_mode = False, use_bn = False, verbose = False, use_deconv=True)
    >>> len(simple_mod.layers)
    7
    >>> simple_mod = build_2d_umodel(np.zeros((1,1,32,32)), 1, [('Chest XRay', 1, 2)], crop_mode = False, use_bn = False, verbose = False, use_deconv=True, use_leakyrelu=0.1)
    >>> len(simple_mod.layers)
    11
    """

    raw_conv_op = lambda n_filters, f_width, name, activation='relu', \
                         border_mode='same', **kwargs: Convolution2D(n_filters,
                                                                     (f_width,
                                                                      f_width),
                                                                     strides=(
                                                                         1, 1),
                                                                     activation=activation,
                                                                     padding=border_mode,
                                                                     name=fix_name_tf(
                                                                         name),
                                                                     **kwargs)
    if use_leakyrelu > 0:
        conv_op = lambda n_filters, f_width, name, activation='relu', \
                         border_mode='same', **kwargs: \
            lambda *args: LeakyReLU(use_leakyrelu)(
                raw_conv_op(n_filters, f_width,
                            name,
                            activation='linear')(
                    *args))
    else:
        conv_op = raw_conv_op

    pool_op = lambda p_size, name, **kwargs: MaxPooling2D(
        pool_size=(p_size, p_size), name=fix_name_tf(name), **kwargs)

    upscale_op = lambda n_filters, p_size, name, **kwargs: UpSampling2D(size=(
        p_size, p_size), name=fix_name_tf(name), **kwargs)
    if use_deconv:
        raw_upscale_op = lambda n_filters, p_size, name, activation='relu',\
                                **kwargs: \
            Conv2DTranspose(
                n_filters,
                kernel_size=(2 * p_size, 2 * p_size),
                strides=(p_size, p_size),
                padding='same',
                name=fix_name_tf(
                    name),
                data_format=K.image_data_format(),
                activation=activation,
                **kwargs)
        if use_leakyrelu > 0:
            upscale_op = lambda n_filters, p_size, name, **kwargs: \
                lambda *args: LeakyReLU(use_leakyrelu)(
                    raw_upscale_op(n_filters, p_size, name,
                                   activation='linear', **kwargs)(
                        *args))
        else:
            upscale_op = raw_upscale_op

    crop_op = lambda crop_d, name, **kwargs: Cropping2D(
        cropping=((crop_d, crop_d), (crop_d, crop_d)),
        name=fix_name_tf(name))

    return _build_nd_umodel(in_img.shape[2:],
                            layers,
                            branches=branches,
                            conv_op=conv_op,
                            pool_op=pool_op,
                            upscale_op=upscale_op,
                            crop_op=crop_op,
                            layer_size_fcn=lsf,
                            pool_size=pool_size,
                            dropout_rate=dropout_rate,
                            last_layer_depth=last_layer_depth,
                            use_bn=use_bn,
                            use_b_conv=use_b_conv,
                            crop_mode=crop_mode,
                            single_input=single_input,
                            verbose=verbose,
                            use_deconv=use_deconv,
                            input_dict=input_dict)


if __name__ == "__main__":
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import petnet

    doctest.testmod(petnet, verbose=True)
