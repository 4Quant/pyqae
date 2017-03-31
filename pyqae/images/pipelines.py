__doc__ = """
The pipelines toolset is designed to make using image processing much easier to combine with sklearn
"""
import numpy as np
from skimage.measure import regionprops
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.pipeline import Pipeline


class ParameterFreeTransform(object):
    def get_params(self, deep):
        return dict()

    def set_params(self):
        raise NotImplementedError('Pipe has no parameters to set!')


class ImmutablePipeTransform(ParameterFreeTransform):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """

    def __init__(self, step_func):
        self._step_func = step_func

    def fit(self, *args):
        return self

    def transform(self, X):
        return self._step_func(X)


def flatten_transform():
    """
    Flatten an image to a vector
    :return:
    >>> p = Pipeline([('flatten',flatten_transform()),('knn',KNC(1))])
    >>> new_p = p.fit(np.arange(18).reshape((2,3,3)),[1,2])
    >>> new_p.predict(np.arange(18).reshape((2,3,3)))
    array([1, 2])
    >>> new_p.predict(10+np.arange(18).reshape((2,3,3)))
    array([2, 2])
    """
    return ImmutablePipeTransform(
        lambda in_tensor: in_tensor.reshape((in_tensor.shape[0], -1))
    )


def normalize_transform():
    """
    Normalize an image
    :return:
    >>> p = Pipeline([('normalize',normalize_transform())])
    >>> new_p = p.fit(np.zeros((5,3,3)),np.ones((5,3,3)))
    >>> new_p.transform(np.arange(18).reshape((2,3,3))).astype(int)
    array([[[-1, -1, -1],
            [-1,  0,  0],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  1],
            [ 1,  1,  1]]])
    >>> p2 = Pipeline([('normalize',normalize_transform()),('flatten', flatten_transform()),('knn', KNC(1))])
    >>> new_p2 = p2.fit(np.arange(18).reshape((2,3,3)),[1,2])
    >>> new_p2.predict(10+np.arange(18).reshape((2,3,3)))
    array([1, 2])
    """
    return ImmutablePipeTransform(
        lambda in_image: (in_image - in_image.mean()) / in_image.std()
    )


from pyqae.nd import meshgridnd_like

identity_transformer=ImmutablePipeTransform(lambda x: x)

class WrappedChannelTransform(object):
    """
    Wrapper for turning transforms into image channel transforms
    The input is expected to be a
        sample, ch, x, y
        sample, ch, x, y, z
    or ch at the end if tf is true
    >>> wct=WrappedChannelTransform(identity_transformer,'tf')
    >>> _ = wct.fit(np.zeros((1,2,3,4)),np.zeros((1,2,3,4)))
    >>> out_arr=wct.transform(np.arange(24).reshape((1,2,3,4)))
    >>> out_arr.shape
    (1, 2, 3, 4)
    >>> out_arr
    array([[[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    <BLANKLINE>
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]]])
    >>> wct=WrappedChannelTransform(identity_transformer,'th')
    >>> _ = wct.fit(np.zeros((1,2,3,4)),np.zeros((1,2,3,4)))
    >>> out_arr=wct.transform(np.arange(24).reshape((1,2,3,4)))
    >>> out_arr.shape
    (1, 2, 3, 4)
    >>> out_arr
    array([[[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    <BLANKLINE>
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]]])
    >>> from sklearn.decomposition import PCA
    >>> wpca=WrappedChannelTransform(PCA(1,random_state=123),'th')
    >>> _ = wpca.fit(np.arange(24).reshape((1,2,3,4)),np.zeros((1,2,3,4)))
    >>> out_arr=wpca.transform(np.arange(24).reshape((1,2,3,4)))
    >>> out_arr.shape
    (1, 1, 3, 4)
    >>> out_arr.astype(int)
    array([[[[-7, -6, -4, -3],
             [-2,  0,  0,  2],
             [ 3,  4,  6,  7]]]])
    """

    def __init__(self,
                 transformer,
                 dim_order, # type: Optional[str]
                 ):
        if dim_order is None:
            # use keras to guide us
            import keras.backend as K
            self.dim_order = K.image_dim_ordering()
        else:
            self.dim_order = dim_order
        self._transformer = transformer

    def get_params(self, *args, **kwargs):
        return self._transformer.get_params(*args,**kwargs)

    def set_params(self, *args, **kwargs):
        return self._transformer.set_params(*args, **kwargs)

    def fit(self, X, y, **kwargs):
        x_vec=self._flatten_data(X)
        y_vec=self._flatten_data(y)
        self._transformer=self._transformer.fit(x_vec, y_vec)
        return self

    def _unflatten_image(self, x_vec, in_shape):
        vec_samples,vec_features=x_vec.shape
        n_samples=in_shape[0]
        x_img=x_vec.reshape((n_samples,-1,vec_features))
        if self.dim_order is 'tf':
            # x, y, ch
            im_shape=tuple(in_shape[1:-1])
        elif self.dim_order is 'th':
            # ch, x, y
            im_shape = tuple(in_shape[2:])
        else:
            raise ValueError('Dim Order not supported: {}'.format(
                self.dim_order))

        x_img = x_img.reshape((n_samples,) + im_shape + (vec_features,))

        if self.dim_order is 'tf':
            return x_img
        elif self.dim_order is 'th':
            return x_img.transpose([0, len(x_img.shape) - 1] +
                                   list(range(1, len(x_img.shape)-1)))

    def _flatten_data(self, X):
        return np.concatenate([self._flatten_image(x_img) for x_img in X],0)

    def _flatten_image(self, x_img):
        assert len(x_img.shape) > 2, "Requires at least a 2D channel image"
        if self.dim_order is 'tf':
            # x, y, ch
            n_img = np.rollaxis(x_img, -1)
        elif self.dim_order is 'th':
            # ch, x, y
            n_img=x_img
        else:
            raise ValueError('Dim Order not supported: {}'.format(
                self.dim_order))
        return n_img.reshape((n_img.shape[0], -1)).swapaxes(0,1)


    def transform(self, X):
        x_vec = self._flatten_data(X)
        return self._unflatten_image(self._transformer.transform(x_vec),
                                     X.shape)



class ChannelPipeTransform(ParameterFreeTransform):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    The input is expected to be a
        sample, ch, x, y
        sample, ch, x, y, z
    or ch at the end if tf is true
    The channel_fcn should accept functions in the ch, x, y... format and
    return them in the same format (functions that only return images (x,
    y) require the np.expand_dims (see distance_transform)
    """

    def __init__(self,
                 channel_fcn, # type:
                 use_generator, # type: bool
                 dim_order, # type: Optional[str]
                 fit_fcn = lambda x,y: None
                 ):
        if dim_order is None:
            # use keras to guide us
            import keras.backend as K
            self.dim_order = K.image_dim_ordering()
        else:
            self.dim_order = dim_order
        self.use_generator = use_generator
        self._channel_fcn = channel_fcn
        self._fit_params = None
        self._fit_fcn = fit_fcn

    def fit(self, *args):
        self._fit_params = self._fit_fcn(*args)
        return self

    def _apply_channel_func(self, X):
        for x_img in X:
            assert len(x_img.shape) > 2, "Requires at least a 2D channel image"
            if self.dim_order is 'tf':
                # x, y, ch
                if len(x_img.shape) == 3:
                    pos_channels = self._channel_fcn(np.rollaxis(x_img, -1))
                elif len(x_img.shape) == 4:
                    pos_channels = self._channel_fcn(np.rollaxis(x_img, -1))
                else:
                    raise ValueError("Above 4D images are not supported {"
                                     "}".format(x_img.shape))
                npos_channels = [np.expand_dims(cpos, -1) for cpos in
                                 pos_channels]
                yield np.concatenate(tuple(npos_channels) + (x_img,), -1)
            elif self.dim_order is 'th':
                # ch, x, y
                if len(x_img.shape) == 3:
                    pos_channels = self._channel_fcn(x_img)
                elif len(x_img.shape) == 4:
                    pos_channels = self._channel_fcn(x_img)
                else:
                    raise ValueError("Above 4D images are not supported {"
                                     "}".format(x_img.shape))
                npos_channels = [np.expand_dims(cpos, 0) for cpos in
                                 pos_channels]
                yield np.concatenate(tuple(npos_channels) + (x_img,), 0)
            else:
                raise ValueError('Dim Order not supported: {}'.format(
                    self.dim_order))

    def transform(self, X):
        res_gen = self._apply_channel_func(X)
        if self.use_generator:
            return res_gen
        else:
            # turn the generator back into a numpy array
            return np.stack(list(res_gen), 0)


def add_position_transform(use_generator=False, dim_order='tf'):
    """
        Adds XX, YY to image data,
    The input is expected to be a
        sample, ch, x, y
        sample, ch, x, y, z
    or ch at the end if tf is true
    :param generator:
    :param dim_order:
    :return:
    >>> p = Pipeline([('add_pos',add_position_transform(True,'th'))])
    >>> new_p = p.fit(np.zeros((1,1,2,2)),np.ones((1,1,2,2)))
    >>> [x.shape for x in new_p.transform(np.arange(8).reshape((2,1,2,2)))]
    [(3, 2, 2), (3, 2, 2)]
    >>> p = Pipeline([('add_pos',add_position_transform(True,'tf'))])
    >>> new_p = p.fit(np.zeros((1,2,2,1)),np.ones((1,2,2,1)))
    >>> [x.shape for x in new_p.transform(np.arange(8).reshape((2,2,2,1)))]
    [(2, 2, 3), (2, 2, 3)]
    >>> p = Pipeline([('add_pos',add_position_transform(False,'th'))])
    >>> new_p = p.fit(np.zeros((1,2,2,1)),np.ones((1,2,2,1)))
    >>> new_p.transform(np.arange(4).reshape((1,1,2,2))).astype(int)
    array([[[[0, 0],
             [1, 1]],
    <BLANKLINE>
            [[0, 1],
             [0, 1]],
    <BLANKLINE>
            [[0, 1],
             [2, 3]]]])
    """
    chan_fcn = lambda x_img: meshgridnd_like(x_img[0])
    return ChannelPipeTransform(chan_fcn, use_generator, dim_order)


from scipy.ndimage.morphology import distance_transform_edt as distmap


def add_distance_transform(use_generator=False, dim_order='tf', img_channel=0):
    """
    Add a distance transform
    :param use_generator:
    :param dim_order:
    :return:
    >>> p = Pipeline([('add_dm',add_distance_transform(True,'th'))])
    >>> new_p = p.fit(np.zeros((1,1,2,2)),np.ones((1,1,2,2)))
    >>> [x.shape for x in new_p.transform(np.arange(18).reshape((2,1,3,3)))]
    [(2, 3, 3), (2, 3, 3)]
    >>> p2 = Pipeline([('add_dm',add_distance_transform(False,'tf'))])
    >>> new_p2 = p2.fit(np.zeros((1,2,2,1)),np.ones((1,2,2,1)))
    >>> t_img = np.pad(np.ones((3,3)),(1,1),'constant', constant_values=0)
    >>> out_p2 = new_p2.transform([np.expand_dims(t_img,-1)])
    >>> out_p2.shape
    (1, 5, 5, 2)
    >>> out_p2[0,:,:,0].astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 2, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    """
    chan_fcn = lambda x_img: np.expand_dims(distmap(x_img[img_channel]), 0)
    return ChannelPipeTransform(chan_fcn, use_generator, dim_order)


from skimage.filters import thresholding as thresh


class TrainableThresholdPipeTransform(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    >>> otsu = TrainableThresholdPipeTransform(thresh.threshold_otsu,'otsu')
    >>> p = Pipeline([('threshold',otsu)])
    >>> new_p = p.fit(np.arange(18).reshape((2,3,3)),np.ones((5,3,3)))
    >>> new_p.transform(np.arange(18).reshape((2,3,3))-4)
    array([[[False, False, False],
            [False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False,  True,  True],
            [ True,  True,  True]]], dtype=bool)
    """

    def __init__(self, thresh_func, thresh_name):
        self._thresh_func = thresh_func
        self._thresh_name = thresh_name
        self.t_val = None

    def fit(self, X, *args):
        self.t_val = self._thresh_func(X)
        return self

    def transform(self, X):
        if self.t_val is None:
            raise NotImplementedError('{} : Step must be fit first'.format(
                self._thresh_name))
        return X > self.t_val


thresh_step = ImmutablePipeTransform(lambda img: (img > 0.5).astype(np.uint8))
shape_step = ImmutablePipeTransform(
    lambda img_list: [shape_analysis(img) for img in img_list])
feature_step = ImmutablePipeTransform(lambda shape_list: np.vstack(
    [np.array(list(shape_dict.values())).reshape(1, -1)
     for shape_dict in shape_list]))


def shape_analysis(in_label):
    try:
        mean_anisotropy = np.mean([(
                                   freg.major_axis_length - freg.minor_axis_length) / freg.minor_axis_length
                                   for freg in regionprops(in_label)])
    except ZeroDivisionError:
        mean_anisotropy = 0
    return dict(
        total_area=np.sum([freg.area for freg in regionprops(in_label)]),
        total_perimeter=np.sum(
            [freg.perimeter for freg in regionprops(in_label)]),
        mean_anisotropy=mean_anisotropy,
        mean_orientation=np.mean(
            [freg.orientation for freg in regionprops(in_label)])
    )


def simple_pipeline_demo():
    return Pipeline([
        ('norm_image', normalize_transform),
        ('threshold', thresh_step),
        ('shape_analysis', shape_step),
        ('feature', feature_step),
        ('KNN', KNC(1))
        # use just the 1st one (more is often better)
    ])


import matplotlib.pyplot as plt
from skimage.util.montage import montage2d

def visualize_pipeline(c_pipe, input_data, verbose = False):
    # type: (Pipeline,  np.ndarray) -> plt.Figure
    """
    Show the full pipeline and results
    :param c_pipe:
    :param input_image:
    :return:
    >>> t_knc = KNC(1)
    >>> p2 = Pipeline([('add_dm',add_distance_transform(False,'tf')),('Flatten',flatten_transform()),('KNC', t_knc)])
    >>> t_img = np.pad(np.ones((3,3)),(1,1),'constant', constant_values=0)
    >>> nt_img = np.expand_dims(np.expand_dims(t_img,-1),0)
    >>> new_p2 = p2.fit(nt_img,[1])
    >>> _ = visualize_pipeline(p2,np.ones((2, 5, 5, 1)),True)
    add_dm - (1, 5, 5, 2)
    Flatten - (1, 50)
    KNC - (1, 1)
    add_dm - (1, 5, 5, 2)
    Flatten - (1, 50)
    KNC - (1, 1)
    >>> _ = visualize_pipeline(p2,np.ones((1, 5, 5, 1)),True)
    add_dm - (1, 5, 5, 2)
    Flatten - (1, 50)
    KNC - (1, 1)
    """
    # focus the training on the non-zero elements (the empty ones dont help us much, for the real data, we cant ignore them though)
    im_settings = {'vmin': 0, 'vmax': 1.5, 'cmap': 'RdBu',
                   'interpolation': 'bicubic'}

    steps = c_pipe.steps
    fig, m_axes = plt.subplots(len(input_data), len(steps) + 1,
                               figsize=(44, 14))
    m_axes = np.expand_dims(m_axes,0) if len(m_axes.shape)==1 else m_axes
    # collect all outputs
    for i, (c_img,c_ax) in enumerate(zip(input_data, m_axes)):
        cur_variable = np.expand_dims(c_img,0)
        c_ax[0].imshow(c_img[0], cmap='bone')
        c_ax[0].set_title('Image {}'.format(i))
        c_ax[0].axis('off')
        for lay_idx, (c_subax, (step_name, step_tool)) in \
                enumerate(zip(c_ax[1:],steps)):

            if hasattr(step_tool,'transform'):
                cur_variable = step_tool.transform(cur_variable)
            elif hasattr(step_tool,'predict_proba'):
                cur_variable = step_tool.predict_proba(cur_variable)
            else:
                cur_variable = step_tool.predict(cur_variable)
            if verbose: print('{} - {}'.format(step_name,cur_variable.shape))
            c_mat  = cur_variable[0,0] if cur_variable.shape[1] == 1 else \
                cur_variable[0]

            if len(c_mat.shape) == 1:
                ind = np.array(range(len(c_mat)))
                c_subax.bar(ind, c_mat)
            elif len(c_mat.shape) == 2:
                c_subax.imshow(c_mat, **im_settings)
            elif len(c_mat.shape) == 3:
                c_subax.imshow(montage2d(c_mat), **im_settings)
            elif len(c_mat.shape) == 4:
                c_subax.imshow(montage2d(
                    np.stack([montage2d(c_layer) for c_layer in c_mat], 0)),
                            **im_settings)
            c_subax.axis('off')
            c_subax.set_title(step_name)
    return fig

class JunkImmutablePipeTransformer(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """

    def __init__(self, step_func, step_name):
        self._step_func = step_func
        self._step_name = step_name

    def fit(self, *args):
        raise NotImplementedError('{}: Once fit this operation cannot be '
                                  're-fit'.format(self._step_name))

    def transform(self, X):
        return self._step_func(X)


if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.images import pipelines

    doctest.testmod(pipelines, verbose=True, optionflags=doctest.ELLIPSIS)
