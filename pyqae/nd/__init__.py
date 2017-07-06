# Tools for ND data processing 
from bolt.spark.array import BoltArraySpark as raw_array
from bolt.spark.construct import ConstructSpark as cs

from pyqae.backend import Row, RDD

sp_array = cs.array
import numpy as np
from numpy import ndarray
from numpy import stack
import os
from skimage.io import imsave
import warnings
from pyqae.utils import Optional, List, Tuple, pprint # noinspection PyUnresolvedReferences

def in_nd(x,y):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    A simple wrapper for the in1d function to work on ND data
    :param x:
    :param y:
    :return:
    >>> t_img = np.arange(6).reshape((2,3))
    >>> pprint(t_img)
    [[0 1 2]
     [3 4 5]]
    >>> pprint(in_nd(t_img, [4,5]))
    [[False False False]
     [False  True  True]]
    """
    return np.in1d(x.ravel(),y).reshape(x.shape)

def meshgridnd_like(in_img,
                    rng_func=range):
    """
    Makes a n-d meshgrid in the shape of the input image
    >>> import numpy as np
    >>> xx, yy = meshgridnd_like(np.ones((3,2)))
    >>> xx.shape
    (3, 2)
    >>> xx
    array([[0, 0],
           [1, 1],
           [2, 2]])
    >>> xx[:,0]
    array([0, 1, 2])
    >>> yy
    array([[0, 1],
           [0, 1],
           [0, 1]])
    >>> yy[0,:]
    array([0, 1])
    >>> xx, yy, zz = meshgridnd_like(np.ones((2,3,4)))
    >>> xx.shape
    (2, 3, 4)
    >>> xx[:,0,0]
    array([0, 1])
    >>> yy[0,:,0]
    array([0, 1, 2])
    >>> zz[0,0,:]
    array([0, 1, 2, 3])
    >>> zz.astype(int)
    array([[[0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]],
    <BLANKLINE>
           [[0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]]])
    """
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])


def filt_tensor(image_stack,  # type: ndarray
                filter_op,  # type: Callable[np.ndarray, np.ndarray]
                tile_size=(128, 128),
                padding=(0, 0),
                *filter_args,
                **filter_kwargs):
    """
    Run an operation on a 4D tensor object (image_number, width, height, channel)

    Parameters
    ----------
    image_stack : ndarray
        a 4D distributed image
    filter_op : (ndarray) -> ndarray
        the operation to apply to 2D slices (width x height) for each channel
    tile_size : (int, int)
        the size of the tiles to cut
    padding : (int, int)
        the padding around the border
    filter_args :
        arguments to pass to the filter_op
    filter_kwargs :
        keyword arguments to pass to filter_op
    :return: a 4D tensor object of the same size
    """

    assert len(
        image_stack.shape) == 4, "Operations are intended for 4D inputs, {}".format(
        image_stack.shape)
    assert isinstance(image_stack,
                      raw_array), "Can only be performed on BoltArray Objects"
    chunk_image = image_stack.chunk(size=tile_size, padding=padding)
    filt_img = chunk_image.map(
        lambda col_img: stack(
            [filter_op(x, *filter_args, **filter_kwargs) for x in
             col_img.swapaxes(0, 2)],
            0).swapaxes(0, 2))
    return filt_img.unchunk()


def tensor_from_rdd(in_rdd,  # type: RDD
                    extract_array=lambda x: x[1],  # type: Callable[Any, ndarray]
                    sort_func=None,  # type: Optional[Callable[Any,int]]
                    make_idx=None  # type: Optional[Callable[Any,int]]
                    ):
    """
    Create a tensor object from an RDD of images

    :param in_rdd: the RDD to process
    :param extract_array: the function to extract the array from the rdd (typically take the value, but some cases (ie dataframe) require more work
    :param sort_func: the function to sort by (leave none for no sorting)
    :param make_idx: the function to use to make the index otherwise just keep the sort
    :return: a distributed ND array containing the full tensor
    """
    in_rdd = in_rdd if sort_func is None else in_rdd.sortBy(sort_func)
    zip_rdd = in_rdd.zipWithIndex().map(lambda x: (
    x[1], extract_array(x[0]))) if make_idx is None else in_rdd.map(
        lambda x: (make_idx(x), extract_array(x)))
    key, val = zip_rdd.first()
    if len(val.shape) == 2:
        add_channels = lambda x: np.expand_dims(x, 2)
        zip_rdd = zip_rdd.mapValues(add_channels)  # add channel
        val = add_channels(val)
    return fromrdd(zip_rdd, dims=val.shape, dtype=val.dtype)


def fromrdd(rdd,
            dims=None,  # type: (int, int, int)
            nrecords=None,  # type: Optional[int]
            dtype=None,  # type: Optional[np.dtype]
            ordered=False
            ):
    """
    Load images from a RDD.
    Input RDD must be a collection of key-value pairs
    where keys are singleton tuples indexing images,
    and values are 2d or 3d ndarrays.

    Parameters
    ----------
    rdd : SparkRDD
        An RDD containing the images.
    dims : tuple or array, optional, default = None
        Image dimensions (if provided will avoid check).
    nrecords : int, optional, default = None
        Number of images (if provided will avoid check).
    dtype : string, default = None
        Data numerical type (if provided will avoid check)
    ordered : boolean, optional, default = False
        Whether or not the rdd is ordered by key
    """

    if dims is None or dtype is None:
        item = rdd.values().first()
        dtype = item.dtype
        dims = item.shape

    if nrecords is None:
        nrecords = rdd.count()

    def process_keys(record):
        k, v = record
        if isinstance(k, int):
            k = (k,)
        return k, v

    return raw_array(rdd.map(process_keys), shape=(nrecords,) + tuple(dims),
                     dtype=dtype, split=1, ordered=ordered)


def save_tensor_local(in_bolt_array, base_path, allow_overwrite=False,
                      file_ext='tif'):
    """
    Save a bolt_array on local (shared directory) paths
    :param in_bolt_array: the bolt array object
    :param base_path: the directory to save in
    :param allow_overwrite: allow overwrite to occur
    :param file_ext: the extension to save to
    """
    try:
        os.mkdir(base_path)
    except:
        print('{} already exists'.format(base_path))
        if not allow_overwrite: raise RuntimeError(
            "Overwriting has not been enabled! Please remove directory {}".format(
                base_path))
    key_fix = lambda in_keys: "_".join(map(lambda k: "%05d" % (k), in_keys))

    in_bolt_array.tordd().map(
        lambda x: imsave(
            os.path.join(base_path, "{}.{}".format(key_fix(x[0]), file_ext)),
            x[1])).collect()
    return base_path


def tensor_to_dataframe(in_bolt_array):
    """
    Create a Spark DataFrame from a BoltArray
    :param in_bolt_array:  the input bolt array
    :return: a DataFrame with fields `position` and `array_data`
    """
    return in_bolt_array.tordd().map(
        lambda x: Row(position=x[0], array_data=x[1].tolist())).toDF()


def save_tensor_parquet(in_bolt_array, out_path):
    return tensor_to_dataframe(in_bolt_array).write.parquet(out_path)


def _dsum(carr,  # type: np.ndarray
          cax  # type: int
          ):
    # type: (...) -> np.ndarray
    """
    Sums the values along all other axes but the current
    :param carr:
    :param cax:
    :return:

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> _dsum(np.zeros((3,3)), 0).astype(np.int8)
    array([0, 0, 0], dtype=int8)
    >>> _dsum(np.eye(3), 1).astype(np.int8)
    array([1, 1, 1], dtype=int8)
    >>> _dsum(np.random.randint(0, 5, size = (3,3,3)), 0).astype(np.int8)
    array([19, 17, 19], dtype=int8)
    """
    return np.sum(carr, tuple(n for n in range(carr.ndim) if n is not cax))


def get_bbox(in_vol,
             min_val=0):
    # type: (np.ndarray, float) -> List[Tuple[int,int]]
    """
    Calculate a bounding box around an image in every direction
    :param in_vol: the array to look at
    :param min_val: the value it must be greater than to add (not equal)
    :return: a list of min,max pairs for each dimension

    >>> import numpy as np
    >>> get_bbox(np.zeros((3,3)), 0)
    [(0, 0), (0, 0)]
    >>> get_bbox(np.pad(np.eye(3).astype(np.int8), 1, mode = lambda *args: 0))
    [(1, 4), (1, 4)]
    """
    ax_slice = []
    for i in range(in_vol.ndim):
        c_dim_sum = _dsum(in_vol > min_val, i)
        wh_idx = np.where(c_dim_sum)[0]
        c_sl = sorted(wh_idx)
        if len(wh_idx) == 0:
            ax_slice += [(0, 0)]
        else:
            ax_slice += [(c_sl[0], c_sl[-1] + 1)]
    return ax_slice


def apply_bbox(in_vol,  # type: np.ndarray
               bbox_list,  # type: List[Tuple[int,int]]
               pad_values=False,
               padding_mode='edge'
               ):
    # type: (...) -> np.ndarray
    """
    Apply a bounding box to an image
    :param in_vol:
    :param bbox_list:
    :param pad_values: expand the image so that negative regions show up as
    well (the size of the output is max-min), default is the same as numpy
    :param padding_mode: the mode to use, see numpy.pad
    :return:
    >>> apply_bbox(np.eye(4).astype(np.uint8), [(-5,2), (0,2)])
    array([[1, 0],
           [0, 1]], dtype=uint8)
    >>> warnings.filterwarnings('ignore') # not finished yet
    >>> apply_bbox(np.eye(4).astype(np.uint8), [(-2,2), (0,2)], True)
    array([[1, 0],
           [1, 0],
           [1, 0],
           [0, 1]], dtype=uint8)
    >>> apply_bbox(np.eye(4).astype(np.uint8), [(0,2), (-2,2)], True).shape
    (2, 4)
    """

    if pad_values:
        # TODO test padding
        warnings.warn("Padded apply_bbox not fully tested yet", RuntimeWarning)
        n_pads = []  # type: List[Tuple[int,int]]
        n_bbox = []  # type: List[Tuple[int, int]]
        for dim_idx, ((a, b), dim_size) in enumerate(zip(bbox_list,
                                                         in_vol.shape)):
            a_pad = 0 if a >= 0 else -a
            b_pad = 0 if b < dim_size else b - dim_size + 1
            n_pads += [(a_pad, b_pad)]
            n_bbox += [(a + a_pad, b + a_pad)]   # adjust the box
        # update the volume
        in_vol = np.pad(in_vol, n_pads, mode=padding_mode)
        # update the bounding box list
        bbox_list = n_bbox

    return in_vol.__getitem__([slice(a, b, 1) for (a, b) in bbox_list])


def autocrop(in_vol,  # type: np.ndarray
             min_val  # type: double
             ):
    # type (...) -> np.ndarray
    """
    Perform an autocrop on an image by keeping all the points above a value
    :param in_vol:
    :param min_val:
    :return:

    >>> np.random.seed(1234)
    >>> autocrop(np.zeros((3,3)), 0).astype(np.bool)
    array([], shape=(0, 0), dtype=bool)
    >>> autocrop(np.eye(3), 0).astype(np.bool)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)
    >>> autocrop(np.random.randint(0, 10, size = (4,4)), 8).astype(np.int8)
    array([[8, 9, 1],
           [9, 6, 8],
           [5, 0, 9]], dtype=int8)
    >>> autocrop(np.pad(np.eye(3).astype(np.int8), 1, mode = lambda *args: 0),0)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int8)
    """
    return apply_bbox(in_vol, get_bbox(in_vol,
                                       min_val=min_val))


def stretch_box(min_val, max_val, new_wid):
    # type: (int, int, int) -> Tuple[int, int]
    """
    For changing range widths without moving the center (much)
    :param min_val:
    :param max_val:
    :param new_wid: the new width of the range
    :return: the new minimum and maximum values
    Examples
    ----
    >>> stretch_box(0, 10, 20) # making the range 0-10 have a width of 20
    (-5, 15)
    >>> stretch_box(10, 20, 5) # making the range 10-20 have the width of 5
    (12, 17)
    """
    old_wid = max_val - min_val
    hwid = int(round((new_wid - old_wid) / 2))
    new_min = min_val - hwid
    new_max = new_min + new_wid
    return (new_min, new_max)


def stretch_bbox(in_coords, new_widths):
    # type: (List[Tuple[int, int]], List[int]) -> List[Tuple[int, int]]
    """
    Stretch a bounding box using the stretch_box function so it has a given
    size
    :param in_coords:
    :param new_widths:
    :return:
    Examples
    ----
    >>> test_box = [(261, 277), (215, 252), (203, 245)]
    >>> stretch_bbox(test_box, [10, 10, 10])
    [(264, 274), (229, 239), (219, 229)]
    """
    return [stretch_box(min_v, max_v, n_wid) for (min_v, max_v), n_wid in
            zip(in_coords, new_widths)]

def replace_labels_with_bbox(in_labels, def_box = [20, 20, 20]):
    # type: (np.ndarray, Union[List[int], bool]) -> np.ndarray
    """
    Replaces all of the labels with a bounding or fixed size box
    :param in_labels: the original label image
    :param def_box: a box size or False
    :return:
    Examples
    >>> test_image = (np.mod(np.arange(9),3)).reshape((3,3))
    >>> replace_labels_with_bbox(test_image, [3,3,3]) #doctest: +NORMALIZE_WHITESPACE
    array([[1, 2, 2],
       [1, 2, 2],
       [1, 2, 2]])
    >>> i_mask = np.mod(np.arange(49),6).reshape((7,7))
    >>> test_image2 = i_mask*(i_mask<3)
    >>> replace_labels_with_bbox(test_image2, [3,3,3]) #doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 2, 2, 2, 0, 0],
       [0, 0, 2, 2, 2, 0, 0],
       [0, 0, 2, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
    >>> replace_labels_with_bbox(test_image2, False) #doctest: +NORMALIZE_WHITESPACE
    array([[2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2]])
    """
    new_labels = np.zeros_like(in_labels)
    for idx in np.unique(in_labels):
        if idx>0:
            c_box = get_bbox(in_labels==idx)
            if def_box:
                c_box = stretch_bbox(c_box, def_box)
            new_labels[[slice(*ibox,1) for ibox in c_box]] = idx # type: ignore
    return new_labels



def pad_box(in_coords, in_shape, pad_width):
    """
    Pad the size of a box (from get_bbox)
    :param in_coords:
    :param in_shape:
    :param pad_width:
    :return:
    Examples
    ------
    >>> test_box = [(261, 277), (215, 252), (203, 245)]
    >>> pad_box(test_box, (440, 512, 512), 1000)
    [(0, 440), (0, 512), (0, 512)]
    >>> pad_box(test_box, (440, 512, 512), 1)
    [(260, 278), (214, 253), (202, 246)]
    """
    return [(int(np.median([0, min_v - pad_width, box_max_v])),
             int(np.median([0, max_v + pad_width, box_max_v]))) for
            (min_v, max_v), box_max_v in zip(in_coords, in_shape)]


from scipy.ndimage import zoom


def iso_image_rescaler(t_array,  # type: np.ndarray
                       gs_arr,  # type: List[float]
                       res_func=lambda x: np.max(x),
                       # type: Callable[np.ndarray,float]
                       order=0,
                       verbose=False,
                       **kwargs):
    # type: (...) -> Tuple[np.ndarray, List[float]]
    new_v_size = res_func(gs_arr)
    scale_f = np.array(gs_arr) / new_v_size
    if verbose: print(gs_arr, '->', new_v_size, ':', scale_f)
    return zoom(t_array, scale_f, order=order, **kwargs), [new_v_size] * 3


def change_resolution_array(in_data,  # type: np.ndarray
                            old_vox_size,  # type: List[float]
                            new_vox_size,  # type: Union[float, np.ndarray]
                            order=0,
                            verbose=False,
                            **kwargs):
    # type: (...) -> Tuple[np.array, List[float]]
    """
    A tool for changing the resolution of an array
    :param in_data:
    :param old_vox_size:
    :param new_vox_size:
    :param order:
    :param verbose:
    :param kwargs:
    :return:

    >>> import numpy as np
    >>> change_resolution_array(np.eye(3).astype(np.int8), [1.0, 1.0], 0.5, order = 0)
    (array([[1, 1, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 1, 1]], dtype=int8), [0.5, 0.5])
    >>> change_resolution_array(np.eye(2).astype(np.int8), [1.0, 1.0], [2.0, 2.0] , order = 0)
    (array([[1]], dtype=int8), [2.0, 2.0])
    """
    if isinstance(new_vox_size, list):
        new_vox_size = np.array(new_vox_size)
    new_v_size = new_vox_size if isinstance(new_vox_size,
                                            np.ndarray) else np.array(
        [new_vox_size] * len(old_vox_size))
    assert len(new_v_size) == len(
        old_vox_size), "Voxel size and old spacing " + \
                       "must have the same size {}, {}".format(new_v_size,
                                                               old_vox_size)
    scale_f = np.array(old_vox_size) / new_v_size
    if verbose: print(old_vox_size, '->', new_v_size, ':', scale_f)
    return zoom(in_data, scale_f, order=order, **kwargs), list(
        new_v_size.tolist())


def uniform_nd_bias_sampler(x_arr, count=1, base_p=0.5, axis=0,
                            cut_off_val=None, ignore_zeros=False):
    """
    A tool for intelligently sampling from biased distributions
    :param x_arr:
    :param count:
    :param base_p: the occurence rate of values which do not meet the cut_off_val
    :param axis:
    :param cut_off_val: should the sum be interpreted directly or as
    :param ignore_zeros: should empty images be ignored (no in-class values)
    :return:
    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> uniform_nd_bias_sampler(np.arange(6),base_p = 0, cut_off_val = 4)
    array([5])
    >>> uniform_nd_bias_sampler(np.arange(6).reshape((3,2,1,1)),base_p = 0, cut_off_val = 7)
    array([2])
    >>> np.random.seed(1234)
    >>> sorted(uniform_nd_bias_sampler(np.arange(10),count = 10, base_p = 0.5, cut_off_val = 8))
    [5, 7, 7, 9, 9, 9, 9, 9, 9, 9]
    >>> np.sum(uniform_nd_bias_sampler(np.arange(10),count = 10000, base_p = 0.5, cut_off_val = 8)==9)
    6673
    """
    c_mat = _dsum(x_arr, axis).astype(np.float32)
    if cut_off_val is not None:
        c_mat = (c_mat > cut_off_val).astype(np.float32)
    if not ignore_zeros:
        assert c_mat.sum() > 0, "Input array is has no values above cutoff {}".format(
            cut_off_val)
    new_p = base_p * np.sum(c_mat) / np.sum(c_mat == 0)
    d_mat = c_mat
    d_mat[c_mat == 0] = new_p
    d_mat /= d_mat.sum()
    return np.random.choice(np.arange(x_arr.shape[axis]), size=count,
                            replace=True, p=d_mat)


if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae import nd

    doctest.testmod(nd, verbose=True, optionflags=doctest.ELLIPSIS)
