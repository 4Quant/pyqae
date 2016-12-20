import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

__doc__ = """
SKLSeg is a package of tools that wraps common scikit-learn functions into image segmentation tools.
"""


class SKLImageSegment(object):
    """
    Base model for doing image segmentation using SKLearn-based models like RandomForestRegressor and GradientBoostingRegression
    """

    def __init__(self, *args, pixel_level=True, ch_ax=1, output_shape=None, **kwargs):

        self._model = self._build_model(*args, **kwargs)
        self.pixel_level = pixel_level
        self.ch_ax = ch_ax
        self.output_shape = output_shape

    def _build_model(self, *args, **kwargs):
        raise ValueError("Model needs to be implemented before the tool can be used")

    @property
    def get_model(self):
        return self._model

    def fit(self, x, y, **kwargs):
        assert x.shape[0] == y.shape[0], "Number of images should match"
        self.output_shape = y.shape[1:]
        n_x = self._transform_image(x)
        n_y = self._transform_image(y)
        self._model.fit(n_x, n_y, **kwargs)
        return self

    def predict(self, x, **kwargs):
        return self.predict_with_shape(x, self.output_shape, **kwargs)

    def predict_match_shape(self, x, **kwargs):
        return self.predict_with_shape(x, (1,) + x.shape[2:], **kwargs)

    def predict_with_shape(self, x, output_shape, **kwargs):
        if (output_shape != self.output_shape) and not self.pixel_level:
            warnings.warn("Shapes do not match and classification is not pixel level requested:{}, available:{}".format(
                output_shape, self.output_shape), RuntimeWarning)
        new_img = self._model.predict(self._transform_image(x), **kwargs)
        return SKLImageSegment._revtransform_image(new_img, output_shape, self.pixel_level)

    @staticmethod
    def _revtransform_image(x, output_shape, pixel_level):
        assert output_shape is not None, "Output shape must first be set (usually in .fit)"
        if pixel_level:
            return x.reshape((-1,) + output_shape)
        else:
            return x.reshape((-1,) + output_shape)

    def _transform_image(self, x):
        if self.pixel_level:
            return x.swapaxes(0, self.ch_ax).reshape((x.shape[self.ch_ax], -1)).swapaxes(0, 1)
        else:
            return x.reshape((x.shape[0], -1))


class RFImageSegment(SKLImageSegment):
    """
    A RandomForest best segmentation algorithm (regression) for images. It handles converting the data from images
    and then back two images and keeping channels together.

    >>> import numpy as np
    >>> test_img = np.arange(2000).reshape((10,2,10,10))
    >>> test_n_img = np.expand_dims(np.sum(test_img,1) % 4,1)
    >>> rfi = RFImageSegment(pixel_level=True)
    >>> _ = rfi.fit(test_img[0:5], test_n_img[0:5])
    >>> rfi.predict(test_img[5:]).shape
    (5, 1, 10, 10)
    >>> rfi2 = RFImageSegment(pixel_level=False)
    >>> _ = rfi2.fit(test_img[0:5], test_n_img[0:5])
    >>> rfi2.predict(test_img[5:]).shape
    (5, 1, 10, 10)
    """

    def _build_model(self, *args, **kwargs):
        return RandomForestRegressor(*args, **kwargs)


class GBImageSegment(SKLImageSegment):
    """
    A GradientBoosting-based segmentation algorithm (regression) for images. It handles converting the data from images
    and then back two images and keeping channels together.
    """

    def _build_model(self, *args, **kwargs):
        return GradientBoostingRegressor(*args, **kwargs)


class ETImageSegment(SKLImageSegment):
    """
    A ExtraTrees-based segmentation algorithm (regression) for images. It handles converting the data from images
    and then back two images and keeping channels together.
    """

    def _build_model(self, *args, **kwargs):
        return ExtraTreesRegressor(*args, **kwargs)


class DTImageSegment(SKLImageSegment):
    """
    A simple DecisionTree-based segmentation algorithm (regression) for images. It handles converting the data from images
    and then back two images and keeping channels together.
    """

    def _build_model(self, *args, **kwargs):
        return DecisionTreeRegressor(*args, **kwargs)


if __name__ == "__main__":
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import sklseg

    doctest.testmod(sklseg, verbose=True)
