# Tools for ND data processing 
from bolt import *
from bolt.spark.construct import ConstructSpark as cs
import bolt
sp_array = cs.array
import numpy as np

def meshgridnd_like(in_img, rng_func = range):
    """
    Makes a n-d meshgrid in the shape of the input image
    """
    new_shape = list(in_img.shape)
    fixed_shape = [new_shape[1], new_shape[0]]+new_shape[2:] if len(new_shape)>=2 else new_shape 
    all_range = [rng_func(i_len) for i_len in fixed_shape]
    return np.meshgrid(*all_range)


def stack(arr_list, axis=0):
    """
    since numpy 1.8.2 does not have the stack command
    """
    assert axis == 0, "Only works for axis 0"
    return np.vstack(map(lambda x: np.expand_dims(x, 0), arr_list))


def filt_tensor(image_stack, filter_op, tile_size=(128, 128), padding=(0, 0), *filter_args, **filter_kwargs):
    assert len(image_stack.shape) == 4, "Operations are intended for 4D inputs, {}".format(image_stack.shape)
    assert isinstance(image_stack, bolt.base.BoltArray), "Can only be performed on BoltArray Objects"
    chunk_image = image_stack.chunk(size=tile_size, padding=padding)
    filt_img = chunk_image.map(
        lambda col_img: stack([filter_op(x, *filter_args, **filter_kwargs) for x in col_img.swapaxes(0, 2)],
                              0).swapaxes(0, 2))
    return filt_img.unchunk()

