import numpy as np
from skimage.segmentation import slic
from pyqae.utils import pprint  # noinspection PyUnresolvedReferences
from warnings import warn

__doc__ = """
Here we package all of the code for super pixel analysis, more for tests 
than real usage
"""

def batch_slic_time(in_batch, # type: np.ndarray
                     time_steps, # type: int
                    channel = None,  # type: Optional[int]
                    include_mask = False,
                    include_channels = True,
                    safety_steps = 1,
                     **slic_args):
    # type: (...) -> np.ndarray
    """
    Takes an input and transforms the results into a time series of
    connected component labels
    :param in_batch:
    :param channel:
    :param time_steps:
    :param slic_args: arguments for the skimage.segmentation.slic call
    :return:
    SLIC Arguments
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.

    >>> s_eye = np.expand_dims(np.expand_dims(np.eye(3),0),-1)
    >>> f = batch_slic_time(s_eye, 4, include_mask = True,include_channels=False)
    >>> pprint(f.squeeze())
    [[[ 1.  1.  0.]
      [ 1.  1.  0.]
      [ 0.  0.  0.]]
    <BLANKLINE>
     [[ 0.  0.  1.]
      [ 0.  0.  1.]
      [ 0.  0.  0.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 1.  1.  0.]]
    <BLANKLINE>
     [[ 0.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0.  1.]]]
    >>> s_eye2 = np.expand_dims(np.expand_dims(np.eye(3)[::-1],0),-1)
    >>> s_full = np.concatenate([s_eye, s_eye2],0)
    >>> f = batch_slic_time(s_full, 4, include_mask = False, include_channels=True)
    >>> pprint(f.squeeze())
    [[[[ 0.  0.  0.]
       [ 0.  1.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  1.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 1.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]
    <BLANKLINE>
      [[ 0.  0.  0.]
       [ 0.  0.  0.]
       [ 0.  0.  0.]]]]
    """
    assert len(in_batch.shape) == 4, "Expected 4D input"
    batch_size, x_wid, y_wid, channels = in_batch.shape
    out_channels = channels if include_channels else 0
    out_channels += 1 if include_mask else 0

    out_batch = np.zeros((batch_size, time_steps, x_wid, y_wid, out_channels),
                         dtype=np.float32)
    for i, c_img in enumerate(in_batch):
        base_img = c_img[..., channel] if channel is not None else c_img

        base_img = np.clip(base_img, -1, 1)
        if np.abs(base_img).max()>1:
            warn('SLIC is not implemented to work correctly with values '
                 'outside of the range -1 - 1', RuntimeWarning)
        c_label = slic(
            base_img,
            n_segments = np.clip(time_steps - safety_steps,1,time_steps),
            multichannel=(channel is None),
            **slic_args
        )

        for j in range(time_steps):

            # include j==0
            c_roi = (c_label == j)
            start_idx = 0
            if include_mask:
                out_batch[i, j, :, :, 0] = c_roi
                start_idx = 1
            if include_channels:
                for i in range(channels):
                    out_batch[i, j, :, :, i+start_idx] = c_roi * c_img[..., i]
    return out_batch


def slic_seg_model(input_shape = (None, None, 2),
                   output_crop = 2,
                   mlp_layers = 5,
                   mlp_depth = 16,
                   dropout_rate = 0.5):
    """
    An full model using SLIC segmentation and then operating using an MLP on the average features
    :param input_shape:
    :param output_crop:
    :return:
    >>> slic_seg_model().summary() # doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    InputImage (InputLayer)          (None, None, None, 2) 0
    ____________________________________________________________________________________________________
    PETNorm (BatchNormalization)     (None, None, None, 2) 8           InputImage[0][0]
    ____________________________________________________________________________________________________
    SlicLayer (Lambda)               (None, 50, None, None 0           InputImage[0][0]
    ____________________________________________________________________________________________________
    SP_Features (Lambda)             (None, 50, 2)         0           PETNorm[0][0]
                                                                       SlicLayer[0][0]
    ____________________________________________________________________________________________________
    FeatureDropout (Dropout)         (None, 50, 2)         0           SP_Features[0][0]
    ____________________________________________________________________________________________________
    TDense_0 (TimeDistributed)       (None, 50, 16)        48          FeatureDropout[0][0]
    ____________________________________________________________________________________________________
    TDense_1 (TimeDistributed)       (None, 50, 16)        272         TDense_0[0][0]
    ____________________________________________________________________________________________________
    TDense_2 (TimeDistributed)       (None, 50, 16)        272         TDense_1[0][0]
    ____________________________________________________________________________________________________
    TDense_3 (TimeDistributed)       (None, 50, 16)        272         TDense_2[0][0]
    ____________________________________________________________________________________________________
    TDense_4 (TimeDistributed)       (None, 50, 16)        272         TDense_3[0][0]
    ____________________________________________________________________________________________________
    TNorm_Out (TimeDistributed)      (None, 50, 1)         17          TDense_4[0][0]
    ____________________________________________________________________________________________________
    Features_to_Label (Lambda)       (None, None, None, No 0           TNorm_Out[0][0]
                                                                       SlicLayer[0][0]
    ____________________________________________________________________________________________________
    FinalCrop (Cropping2D)           (None, None, None, No 0           Features_to_Label[0][0]
    ====================================================================================================
    Total params: 1,161
    Trainable params: 1,157
    Non-trainable params: 4
    ____________________________________________________________________________________________________
    """
    # keep the imports inside so we don't make the superpixels package too
    # heavy
    from keras.models import Model
    from keras.layers import Input, Dense, TimeDistributed, Lambda, \
        BatchNormalization, Dropout
    from pyqae.dnn.layers import label_average_features_tf, features_to_label_tf
    from pyqae.dnn.features import SLICTimeChannel
    from keras.layers import Cropping2D
    # NOTE: not compatible with Dropout or Spatial Dropout (results in NANs)
    input_slice = Input(shape=input_shape,
                        name='InputImage')
    in_slice_norm = BatchNormalization(name='PETNorm')(input_slice)
    slic_layer = SLICTimeChannel(50, name='SlicLayer',
                                 include_mask=True,
                                 include_channels=False,
                                 channel=0,
                                 compactness=0.25)
    in_labels = slic_layer(input_slice)

    sp_features = Lambda(lambda x: label_average_features_tf(x[0], x[1]),
                         name='SP_Features')(
        [in_slice_norm, in_labels])
    out_features = sp_features if dropout_rate==0 else \
        Dropout(dropout_rate,  name =  'FeatureDropout')(
        sp_features)
    for i in range(mlp_layers):
        out_features = TimeDistributed(Dense(mlp_depth, activation='relu'),
                                       name='TDense_{}'.format(i))(
            out_features)
    out_features = TimeDistributed(Dense(1, activation='sigmoid'),
                                   name='TNorm_Out')(out_features)
    out_segs = Lambda(lambda x: features_to_label_tf(x[0], x[1]),
                             name='Features_to_Label')(
        [out_features, in_labels])
    last_layer = Cropping2D(
        cropping=((output_crop, output_crop), (output_crop, output_crop)),
        name='FinalCrop')(out_segs)
    return Model(inputs=[input_slice],
                 outputs=[last_layer], name = 'TestSLICModel')


