# Tools for ND data processing 
from bolt.spark.array import BoltArraySpark as raw_array
from bolt.spark.construct import ConstructSpark as cs
from .. import Row
sp_array = cs.array
import numpy as np
from numpy import stack
import os
from skimage.io import imsave

def meshgridnd_like(in_img, rng_func = range):
    """
    Makes a n-d meshgrid in the shape of the input image
    """
    new_shape = list(in_img.shape)
    fixed_shape = [new_shape[1], new_shape[0]]+new_shape[2:] if len(new_shape)>=2 else new_shape 
    all_range = [rng_func(i_len) for i_len in fixed_shape]
    return np.meshgrid(*all_range)


def filt_tensor(image_stack, filter_op, tile_size=(128, 128), padding=(0, 0), *filter_args, **filter_kwargs):
    """
    Run an operation on a 4D tensor object (image_number, width, height, channel)
    :param image_stack: a 4D distributed image
    :param filter_op: the operation to apply to 2D slices (width x height) for each channel
    :param tile_size: the size of the tiles to cut
    :param padding: the padding around the border
    :param filter_args: arguments to pass to the filter_op
    :param filter_kwargs: keyword arguments to pass to filter_op
    :return: a 4D tensor object of the same size
    """

    assert len(image_stack.shape) == 4, "Operations are intended for 4D inputs, {}".format(image_stack.shape)
    assert isinstance(image_stack, raw_array), "Can only be performed on BoltArray Objects"
    chunk_image = image_stack.chunk(size=tile_size, padding=padding)
    filt_img = chunk_image.map(
        lambda col_img: stack([filter_op(x, *filter_args, **filter_kwargs) for x in col_img.swapaxes(0, 2)],
                              0).swapaxes(0, 2))
    return filt_img.unchunk()


def tensor_from_rdd(in_rdd, extract_array = lambda x: x[1], sort_func = None, make_idx = None):
    """
    Create a tensor object from an RDD of images
    :param in_rdd: the RDD to process
    :param extract_array: the function to extract the array from the rdd (typically take the value, but some cases (ie dataframe) require more work
    :param sort_func: the function to sort by (leave none for no sorting)
    :param make_idx: the function to use to make the index otherwise just keep the sort
    :return: a distributed ND array containing the full tensor
    """
    in_rdd = in_rdd if sort_func is None else in_rdd.sortBy(sort_func)
    zip_rdd = in_rdd.zipWithIndex().map(lambda x: (x[1], extract_array(x[0]))) if make_idx is None else in_rdd.map(lambda x: (make_idx(x), extract_array(x)))
    key, val = zip_rdd.first()
    if len(val.shape)==2:
        add_channels = lambda x: np.expand_dims(x,2)
        zip_rdd = zip_rdd.mapValues(add_channels) # add channel
        val = add_channels(val)
    return fromrdd(zip_rdd, dims = val.shape, dtype = val.dtype)


def fromrdd(rdd, dims=None, nrecords=None, dtype=None, ordered=False):
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

    return raw_array(rdd.map(process_keys), shape=(nrecords,) + tuple(dims), dtype=dtype, split=1, ordered=ordered)



def save_tensor_local(in_bolt_array, base_path, allow_overwrite = False, file_ext = 'tif'):
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
        if not allow_overwrite: raise RuntimeError("Overwriting has not been enabled! Please remove directory {}".format(base_path))
    key_fix = lambda in_keys: "_".join(map(lambda k: "%05d" % (k), in_keys))

    in_bolt_array.tordd().map(lambda x: imsave(os.path.join(base_path, "{}.{}".format(key_fix(x[0]), file_ext)), x[1])).collect()
    return base_path

def tensor_to_dataframe(in_bolt_array):
    """
    Create a Spark DataFrame from a BoltArray
    :param in_bolt_array:  the input bolt array
    :return: a DataFrame with fields `position` and `array_data`
    """
    return in_bolt_array.tordd().map(lambda x: Row(position = x[0], array_data = x[1].tolist())).toDF()

def save_tensor_parquet(in_bolt_array, out_path):
    return tensor_to_dataframe(in_bolt_array).write.parquet(out_path)