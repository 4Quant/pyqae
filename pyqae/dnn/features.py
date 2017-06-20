import numpy as np
from keras.layers import Input, Conv3D, ZeroPadding3D, Cropping3D, add, multiply
from keras.layers import Lambda
from keras.models import Model

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
