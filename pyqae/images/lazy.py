import warnings
from typing import List

import numpy as np
from PIL import Image as PImg
from bolt.spark.chunk import ChunkedArray
from bolt.utils import slicify

from pyqae.backend import RDD


class LazyImageBackend(object):
    """
    Interface for having lazy/partial image loading
    """

    def __init__(self, path):
        self.path = path
        self._im_obj = None  # should not be directly referenced

    def __getstate__(self):
        return (self.path,)

    def __setstate__(self, state):
        self.path = state[0]
        self._im_obj = None

    def open_image(self):
        raise RuntimeError("Not yet implemented")

    def read_region(self, xstart, xend, ystart, yend):
        raise RuntimeError("Not yet implemented")

    def get_tile_xy_size(self):
        raise RuntimeError("Not yet implemented")

    def _get_chan_dim(self):
        """
        the dimension of the output image in channels
        :return: tuple () for 2D, (3, ) for RGB
        """
        return self.read_region(0, 1, 0, 1).shape[2:]

    @property
    def dtype(self):
        return self.read_region(0, 1, 0, 1).dtype

    @property
    def shape(self):
        return self.get_tile_xy_size() + self._get_chan_dim()

    @property
    def ndim(self):
        return len(self.shape)


class LazyImagePillowBackend(LazyImageBackend):
    def open_image(self):
        self._im_obj = self._im_obj if self._im_obj is not None else PImg.open(self.path)
        return self._im_obj

    def read_region(self, xstart, xend, ystart, yend):
        imc = self.open_image().crop((xstart, ystart, xend, yend))
        return np.array(imc).swapaxes(0, 1)

    def get_tile_xy_size(self):
        return self.open_image().size


backends = [LazyImagePillowBackend]  # type: List[LazyImageBackend]

try:
    from osgeo import gdal


    class LazyImageGDALBackend(LazyImageBackend):
        """
        A GDAL-based implementation of the lazy image backend
        """

        def open_image(self):
            # remove the caching since it causes strange issues still
            # self._im_obj = self._im_obj if self._im_obj is not None else self._open_image()
            return self._open_image()

        def _open_image(self):
            return gdal.Open(self.path)

        def read_region(self, xstart, xend, ystart, yend):
            temp_img_obj = self.open_image()
            x_size = xend - xstart
            y_size = yend - ystart
            assert x_size >= 0, "X Size must be greater than 0, {}-{}".format(xstart, xend)
            assert y_size >= 0, "Y Size must be greater than 0, {}-{}".format(ystart, yend)
            out_tile = temp_img_obj.ReadAsArray(int(xstart), int(ystart), int(x_size), int(y_size)).swapaxes(0, 1)
            if out_tile is None:
                raise RuntimeError("Tile Information cannot be read {}-{}, {}-{}".format(xstart, xend, ystart, yend))
            return out_tile

        def get_tile_xy_size(self):
            c_img = self.open_image()
            return c_img.RasterXSize, c_img.RasterYSize

        def _get_chan_dim(self):
            """
            the dimension of the output image in channels
            :return: tuple () for 2D, (3, ) for RGB
            """
            rsc = self.open_image().RasterCount
            return (rsc,) if rsc > 1 else tuple()


    backends += [LazyImageGDALBackend]
except ImportError:
    gdal = None
    warnings.warn("The GDAL Lazy Image Backend requires that the GDAL package is installed", ImportWarning)


class DiskMappedLazyImage(object):
    """
    A lazily read image which behaves as if it were a numpy array, fully serializable
    """

    def __init__(self, path, bckend):
        assert isinstance(path, str), "Path must be a single string, {}".format(path)
        self._bckend = bckend(path)
        assert isinstance(self._bckend, LazyImageBackend), "Instantiated backend mmust be a lazy image"
        self.path = path

    @property
    def shape(self):
        return self._bckend.shape

    @property
    def dtype(self):
        return self._bckend.dtype

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        """
        Get an item from the array through indexing.
        Supports basic indexing with slices and ints, or advanced
        indexing with lists or ndarrays of integers.
        Mixing basic and advanced indexing across axes is currently supported
        only for a single advanced index amidst multiple basic indices.
        Parameters
        ----------
        index : tuple of slices
            One or more index specifications
        Returns
        -------
        NDArray
        """
        if isinstance(index, tuple):
            index = list(index)
        else:
            index = [index]
        int_locs = np.where([isinstance(i, int) for i in index])[0]

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if not all([isinstance(i, (slice, int, list, tuple, np.ndarray)) for i in index]):
            raise ValueError("Each index must either be a slice, int, list, set, or ndarray")

        # fill unspecified axes with full slices
        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        # standardize slices and bounds checking
        for n, idx in enumerate(index):
            size = self.shape[n]
            if isinstance(idx, (slice, int)):
                slc = slicify(idx, size)
                # throw an error if this would lead to an empty dimension in numpy
                if slc.step > 0:
                    minval, maxval = slc.start, slc.stop
                else:
                    minval, maxval = slc.stop, slc.start
                if minval > size - 1 or maxval < 1 or minval >= maxval:
                    raise ValueError("Index {} in dimension {} with shape {} would "
                                     "produce an empty dimension".format(idx, n, size))
                index[n] = slc
            else:
                adjusted = np.array(idx)
                inds = np.where(adjusted < 0)
                adjusted[inds] += size
                if adjusted.min() < 0 or adjusted.max() > size - 1:
                    raise ValueError("Index {} out of bounds in dimension {} with "
                                     "shape {}".format(idx, n, size))
                index[n] = adjusted
        # assume basic indexing
        if all([isinstance(i, slice) for i in index]) and (len(index) <= 3):
            assert len(index) > 1, "Too short of an index"
            assert index[0].start <= index[0].stop, "Indexes cannot be backwards"
            assert index[1].start <= index[1].stop, "Indexes cannot be backwards"
            out_arr = self._bckend.read_region(xstart=int(index[0].start), xend=int(index[0].stop),
                                               ystart=int(index[1].start), yend=int(index[1].stop))
            out_arr = out_arr[::index[0].step, ::index[1].step]
            return out_arr[:, :, index[2]] if len(index) == 3 else out_arr

        else:
            raise NotImplementedError("When mixing basic indexing (slices and int) with "
                                      "with advanced indexing (lists, tuples, and ndarrays), "
                                      "can only have a single advanced index")


def paths_to_tiled_image(paths, context=None, tile_size=(256, 256), padding=(0, 0),
                         backend=backends[-1],
                         skip_chunk=False,
                         **kwargs):
    """
    Create an tiled ND image from a collection of paths
    :param paths: List[str] / RDD[str] a list or RDD of strings containing image paths
    :param context: SparkContext the context to make the RDD from if paths is a list
    :param tile_size: the size of tiles to cut
    :param padding:  the padding to use
    :param backend: The LazyImageBackend to use for reading the image data in (by default uses the last one, GDAL if available)
    :param skip_chunk: For developer use only allows the actual subchunking step to be delayed
    :param kwargs: other arguments for creating the initial RDD
    :return: a ChunkedRDD containing the image data as tiles (use .unchunk to make into a normal RDD)
    """
    path_rdd = paths if isinstance(paths, RDD) else context.parallelize(paths, **kwargs)
    _create_dmzi = lambda fle_path: DiskMappedLazyImage(fle_path, backend)
    in_rdd = path_rdd.zipWithIndex().map(lambda x: (x[1], _create_dmzi(x[0])))
    first_ds = _create_dmzi(path_rdd.first())
    ca_data = ChunkedArray(in_rdd,
                           shape=(path_rdd.count(),) + first_ds.shape,
                           split=1,
                           dtype=first_ds[0, 0].dtype
                           )
    if skip_chunk: return ca_data
    return ca_data._chunk(size=tile_size, axis=None, padding=padding)


from itertools import product
def parallel_tile_image(paths, # type: Union[List[str], RDD[str]]
                         context = None, # type: SparkContext
                         backend=backends[-1], # type: LazyImageBackend
                         tile_w = 512,
                         tile_h = None, # type: Optional[int]
                         **kwargs):
    # type: (...) -> RDD[Tuple(int, int, int), np.ndarray]
    """
    A function to read tiles in in parallel
    :param paths: a list of paths to read from
    :param context: the spark context (or local spark context)
    :param backend: the backend for reading
    :param tile_w: the tile width
    :param tile_h: the tile height
    :param kwargs: arguments for parallelize
    :return:
    """
    path_rdd = paths if isinstance(paths, RDD) else context.parallelize(paths, **kwargs)
    _create_dmzi = lambda fle_path: DiskMappedLazyImage(fle_path, backend)
    img_rdd = path_rdd.zipWithIndex().map(lambda x: (x[1], _create_dmzi(x[0])))
    if tile_h is None:
        tile_h = tile_w
    _, c_img = img_rdd.first()
    start_x = np.arange(0, c_img.shape[0], tile_w)
    start_y = np.arange(0, c_img.shape[1], tile_h)
    start_xy = product(start_x, start_y)
    tile_ij_rdd = context.parallelize(start_xy, **kwargs)
    all_tiles = img_rdd.cartesian(tile_ij_rdd)
    def _clean_tile_order(x):
        (img_id, img_data), (tile_i, tile_j) = x
        return (img_id, tile_i, tile_j), img_data[tile_i:tile_i+tile_w, tile_j:tile_j+tile_h]
    return all_tiles.map(_clean_tile_order)

def single_chunky_image(in_ds, context, tile_size=(256, 256), padding=(0, 0)):
    in_rdd = context.parallelize([((0,), in_ds)])
    return ChunkedArray(in_rdd,
                        shape=(in_rdd.count(),) + in_ds.size,
                        split=1,
                        dtype=in_ds[0, 0].dtype
                        )._chunk(size=tile_size, axis=None, padding=padding)


def stack(arr_list, axis=0):
    """
    since numpy 1.8.2 does not have the stack command
    """
    assert axis == 0, "Only works for axis 0"
    return np.vstack(map(lambda x: np.expand_dims(x, 0), arr_list))
