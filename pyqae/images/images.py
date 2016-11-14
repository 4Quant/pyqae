import logging
from numpy import ndarray, arange, amax, amin, size, asarray, random, prod, \
    apply_along_axis
from itertools import product

from ..base import Data


class Images(Data):
    """
    Collection of images or volumes.

    Backed by an array-like object, including a numpy array
    (for local computation) or a bolt array (for spark computation).

    Attributes
    ----------
    values : array-like
        numpy array or bolt array

    labels : array-like or list
        A set of labels, one per image.
    """
    _metadata = Data._metadata

    def __init__(self, values, labels=None, mode='local'):
        super(Images, self).__init__(values, mode=mode)
        self.labels = labels

    @property
    def baseaxes(self):
        return (0,)

    @property
    def _constructor(self):
        return Images

    def count(self):
        """
        Count the number of images.

        For lazy or distributed data, will force a computation.
        """
        if self.mode == 'local':
            return self.shape[0]

        if self.mode == 'spark':
            return self.tordd().count()

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'local':
            return self.values[0]

        if self.mode == 'spark':
            return self.values.first().toarray()

    def toblocks(self, chunk_size='auto', padding=None):
        """
        Convert to blocks which represent subdivisions of the images data.

        Parameters
        ----------
        chunk_size : str or tuple, size of image chunk used during conversion, default = 'auto'
            String interpreted as memory size (in kilobytes, e.g. '64').
            The exception is the string 'auto'. In spark mode, 'auto' will choose a chunk size to make the
            resulting blocks ~100 MB in size. In local mode, 'auto' will create a single block.
            Tuple of ints interpreted as 'pixels per dimension'.

        padding : tuple or int
            Amount of padding along each dimensions for blocks. If an int, then
            the same amount of padding is used for all dimensions
        """
        from pyqae.blocks.blocks import Blocks
        from pyqae.blocks.local import LocalChunks

        if self.mode == 'spark':
            if chunk_size is 'auto':
                chunk_size = str(max([int(1e5/self.shape[0]), 1]))
            chunks = self.values.chunk(chunk_size, padding=padding).keys_to_values((0,))

        if self.mode == 'local':
            if chunk_size is 'auto':
                chunk_size = self.shape[1:]
            chunks = LocalChunks.chunk(self.values, chunk_size, padding=padding)

        return Blocks(chunks)

    def toseries(self, chunk_size='auto'):
        """
        Converts to series data.

        This method is equivalent to images.toblocks(size).toSeries().

        Parameters
        ----------
        chunk_size : str or tuple, size of image chunk used during conversion, default = 'auto'
            String interpreted as memory size (in kilobytes, e.g. '64').
            The exception is the string 'auto', which will choose a chunk size to make the
            resulting blocks ~100 MB in size. Tuple of ints interpreted as 'pixels per dimension'.
            Only valid in spark mode.
        """
        from thunder.series.series import Series

        if chunk_size is 'auto':
            chunk_size = str(max([int(1e5/self.shape[0]), 1]))

        n = len(self.shape) - 1
        index = arange(self.shape[0])

        if self.mode == 'spark':
            return Series(self.values.swap((0,), tuple(range(n)), size=chunk_size), index=index)

        if self.mode == 'local':
            return Series(self.values.transpose(tuple(range(1, n+1)) + (0,)), index=index)

    def tolocal(self):
        """
        Convert to local mode.
        """
        from pyqae.images.readers import fromarray

        if self.mode == 'local':
            logging.getLogger('pyqae').warn('images already in local mode')
            pass

        return fromarray(self.toarray())

    def tospark(self, engine=None):
        """
        Convert to distributed spark mode.
        """
        from pyqae.images.readers import fromarray

        if self.mode == 'spark':
            logging.getLogger('pyqae').warn('images already in spark mode')
            pass

        if engine is None:
            raise ValueError('Must provide a SparkContext')

        return fromarray(self.toarray(), engine=engine)

    def foreach(self, func):
        """
        Execute a function on each image.

        Functions can have side effects. There is no return value.
        """
        if self.mode == 'spark':
            self.values.tordd().map(lambda kv: (kv[0][0], kv[1])).foreach(func)
        else:
            [func(kv) for kv in enumerate(self.values)]

    def sample(self, nsamples=100, seed=None):
        """
        Extract a random sample of images.

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample.

        seed : int, optional, default = None
            Random seed.
        """
        if nsamples < 1:
            raise ValueError("Number of samples must be larger than 0, got '%g'" % nsamples)

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if self.mode == 'spark':
            result = asarray(self.values.tordd().values().takeSample(False, nsamples, seed))

        else:
            inds = [int(k) for k in random.rand(nsamples) * self.shape[0]]
            result = asarray([self.values[i] for i in inds])

        return self._constructor(result)

    def reduce(self, func):
        """
        Reduce a function over images.

        Parameters
        ----------
        func : function
            A function of two images.
        """
        return self._reduce(func, axis=0)

    def mean(self):
        """
        Compute the mean across images.
        """
        return self._constructor(self.values.mean(axis=0, keepdims=True))

    def var(self):
        """
        Compute the variance across images.
        """
        return self._constructor(self.values.var(axis=0, keepdims=True))

    def std(self):
        """
        Compute the standard deviation across images.
        """
        return self._constructor(self.values.std(axis=0, keepdims=True))

    def sum(self):
        """
        Compute the sum across images.
        """
        return self._constructor(self.values.sum(axis=0, keepdims=True))

    def max(self):
        """
        Compute the max across images.
        """
        return self._constructor(self.values.max(axis=0, keepdims=True))

    def min(self):
        """
        Compute the min across images.
        """
        return self._constructor(self.values.min(axis=0, keepdims=True))

    def squeeze(self):
        """
        Remove single-dimensional axes from images.
        """
        axis = tuple(range(1, len(self.shape) - 1)) if prod(self.shape[1:]) == 1 else None
        return self.map(lambda x: x.squeeze(axis=axis))

    def reshape(self, *shape):
        """
        Reshape images

        Parameters
        ----------
        shape: one or more ints
            New shape
        """
        if prod(self.shape) != prod(shape):
            raise ValueError("Reshaping must leave the number of elements unchanged")

        if self.shape[0] != shape[0]:
            raise ValueError("Reshaping cannot change the number of images")

        if len(shape) not in (3, 4):
            raise ValueError("Reshaping must produce 2d or 3d images")

        return self._constructor(self.values.reshape(shape)).__finalize__(self)

    def max_projection(self, axis=2):
        """
        Compute maximum projections of images along a dimension.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along.
        """
        if axis >= size(self.value_shape):
            raise Exception('Axis for projection (%s) exceeds '
                            'image dimensions (%s-%s)' % (axis, 0, size(self.value_shape)-1))

        new_value_shape = list(self.value_shape)
        del new_value_shape[axis]
        return self.map(lambda x: amax(x, axis), value_shape=new_value_shape)

    def max_min_projection(self, axis=2):
        """
        Compute maximum-minimum projection along a dimension.

        This computes the sum of the maximum and minimum values.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along.
        """
        if axis >= size(self.value_shape):
            raise Exception('Axis for projection (%s) exceeds '
                            'image dimensions (%s-%s)' % (axis, 0, size(self.value_shape)-1))

        new_value_shape = list(self.value_shape)
        del new_value_shape[axis]
        return self.map(lambda x: amax(x, axis) + amin(x, axis), value_shape=new_value_shape)

    def subsample(self, factor):
        """
        Downsample images by an integer factor.

        Parameters
        ----------
        factor : positive int or tuple of positive ints
            Stride to use in subsampling. If a single int is passed,
            each dimension of the image will be downsampled by this factor.
            If a tuple is passed, each dimension will be downsampled by the given factor.
        """
        value_shape = self.value_shape
        ndims = len(value_shape)
        if not hasattr(factor, '__len__'):
            factor = [factor] * ndims
        factor = [int(sf) for sf in factor]

        if any((sf <= 0 for sf in factor)):
            raise ValueError('All sampling factors must be positive; got ' + str(factor))

        def roundup(a, b):
            return (a + b - 1) // b

        slices = [slice(0, value_shape[i], factor[i]) for i in range(ndims)]
        new_value_shape = tuple([roundup(value_shape[i], factor[i]) for i in range(ndims)])

        return self.map(lambda v: v[slices], value_shape=new_value_shape)

    def gaussian_filter(self, sigma=2, order=0):
        """
        Spatially smooth images with a gaussian filter.

        Filtering will be applied to every image in the collection.

        Parameters
        ----------
        sigma : scalar or sequence of scalars, default = 2
            Size of the filter size as standard deviation in pixels.
            A sequence is interpreted as the standard deviation for each axis.
            A single scalar is applied equally to all axes.

        order : choice of 0 / 1 / 2 / 3 or sequence from same set, optional, default = 0
            Order of the gaussian kernel, 0 is a gaussian,
            higher numbers correspond to derivatives of a gaussian.
        """
        from scipy.ndimage.filters import gaussian_filter

        return self.map(lambda v: gaussian_filter(v, sigma, order), value_shape=self.value_shape)

    def uniform_filter(self, size=2):
        """
        Spatially filter images using a uniform filter.

        Filtering will be applied to every image in the collection.

        Parameters
        ----------
        size: int, optional, default = 2
            Size of the filter neighbourhood in pixels.
            A sequence is interpreted as the neighborhood size for each axis.
            A single scalar is applied equally to all axes.
        """
        return self._image_filter(filter='uniform', size=size)

    def median_filter(self, size=2):
        """
        Spatially filter images using a median filter.

        Filtering will be applied to every image in the collection.

        parameters
        ----------
        size: int, optional, default = 2
            Size of the filter neighbourhood in pixels.
            A sequence is interpreted as the neighborhood size for each axis.
            A single scalar is applied equally to all axes.
        """
        return self._image_filter(filter='median', size=size)

    def _image_filter(self, filter=None, size=2):
        """
        Generic function for maping a filtering operation over images.

        Parameters
        ----------
        filter : string
            Which filter to use.

        size : int or tuple
            Size parameter for filter.
        """
        from numpy import isscalar
        from scipy.ndimage.filters import median_filter, uniform_filter

        FILTERS = {
            'median': median_filter,
            'uniform': uniform_filter
        }

        func = FILTERS[filter]

        mode = self.mode
        value_shape = self.value_shape
        ndims = len(value_shape)

        if ndims == 3 and isscalar(size) == 1:
            size = [size, size, size]

        if ndims == 3 and size[2] == 0:
            def filter_(im):
                if mode == 'spark':
                    im.setflags(write=True)
                else:
                    im = im.copy()
                for z in arange(0, value_shape[2]):
                    im[:, :, z] = func(im[:, :, z], size[0:2])
                return im
        else:
            filter_ = lambda x: func(x, size)

        return self.map(lambda v: filter_(v), value_shape=self.value_shape)

    def localcorr(self, size=2):
        """
        Correlate every pixel in an image sequence to the average of its local neighborhood.

        This algorithm computes, for every pixel, the correlation coefficient
        between the sequence of values for that pixel, and the average of all pixels
        in a local neighborhood. It does this by blurring the image(s) with a uniform filter,
        and then correlates the original sequence with the blurred sequence.

        Parameters
        ----------
        size : int or tuple, optional, default = 2
            Size of the filter in pixels. If a scalar, will use the same filter size
            along each dimension.
        """

        from pyqae.images.readers import fromarray, fromrdd
        from numpy import corrcoef, concatenate

        nimages = self.shape[0]

        # spatially average the original image set over the specified neighborhood
        blurred = self.uniform_filter(size)

        # union the averaged images with the originals to create an
        # Images object containing 2N images (where N is the original number of images),
        # ordered such that the first N images are the averaged ones.
        if self.mode == 'spark':
            combined = self.values.concatenate(blurred.values)
            combined_images = fromrdd(combined.tordd())
        else:
            combined = concatenate((self.values, blurred.values), axis=0)
            combined_images = fromarray(combined)

        # correlate the first N (averaged) records with the last N (original) records
        series = combined_images.toseries()
        corr = series.map(lambda x: corrcoef(x[:nimages], x[nimages:])[0, 1])

        return corr.toarray()

    def subtract(self, val):
        """
        Subtract a constant value or an image from all images.

        Parameters
        ----------
        val : int, float, or ndarray
            Value to subtract.
        """
        if isinstance(val, ndarray):
            if val.shape != self.value_shape:
                raise Exception('Cannot subtract image with dimensions %s '
                                'from images with dimension %s' % (str(val.shape), str(self.value_shape)))

        return self.map(lambda x: x - val, value_shape=self.value_shape)

    def topng(self, path, prefix='image', overwrite=False):
        """
        Write 2d images as PNG files.

        Files will be written into a newly-created directory.
        Three-dimensional data will be treated as RGB channels.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from pyqae.images.writers import topng
        # TODO add back colormap and vmin/vmax
        topng(self, path, prefix=prefix, overwrite=overwrite)

    def totif(self, path, prefix='image', overwrite=False):
        """
        Write 2d images as TIF files.

        Files will be written into a newly-created directory.
        Three-dimensional data will be treated as RGB channels.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from pyqae.images.writers import totif
        # TODO add back colormap and vmin/vmax
        totif(self, path, prefix=prefix, overwrite=overwrite)

    def tobinary(self, path, prefix='image', overwrite=False):
        """
        Write out images as flat binary files.

        Files will be written into a newly-created directory.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from pyqae.images.writers import tobinary
        tobinary(self, path, prefix=prefix, overwrite=overwrite)

    def map_as_series(self, func, value_size=None, dtype=None, chunk_size='auto'):
        """
        Efficiently apply a function to images as series data.

        For images data that represent image sequences, this method
        applies a function to each pixel's series, and then returns to
        the images format, using an efficient intermediate block
        representation.

        Parameters
        ----------
        func : function
            Function to apply to each time series. Should take one-dimensional
            ndarray and return the transformed one-dimensional ndarray.

        value_size : int, optional, default = None
            Size of the one-dimensional ndarray resulting from application of
            func. If not supplied, will be automatically inferred for an extra
            computational cost.

        dtype : str, optional, default = None
            dtype of one-dimensional ndarray resulting from application of func.
            If not supplied it will be automatically inferred for an extra computational cost.

        chunk_size : str or tuple, size of image chunk used during conversion, default = 'auto'
            String interpreted as memory size (in kilobytes, e.g. '64').
            The exception is the string 'auto'. In spark mode, 'auto' will choose a chunk size to make the
            resulting blocks ~100 MB in size. In local mode, 'auto' will create a single block.
            Tuple of ints interpreted as 'pixels per dimension'.
        """
        blocks = self.toblocks(chunk_size=chunk_size)

        if value_size is not None:
            dims = list(blocks.blockshape)
            dims[0] = value_size
        else:
            dims = None

        def f(block):
            return apply_along_axis(func, 0, block)

        return blocks.map(f, value_shape=dims, dtype=dtype).toimages()
