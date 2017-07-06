from itertools import product

import numpy as np
from scipy.ndimage import label
from sklearn.feature_extraction.image import extract_patches

from pyqae.utils import pprint  # noinspection PyUnresolvedReferences


def sorted_label(in_img, # type: np.ndarray
                 **kwargs):
    # type: (...) -> np.ndarray
    """
    Return labeled image in a sorted manner by area (in 2D, length in 1D and volume in 3D)
    :param in_img:
    :param kwargs: arguments for the label command
    :return:
    >>> t_img = np.eye(3)
    >>> pprint(sorted_label(t_img))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> t_img[1,2] = 1
    >>> pprint(sorted_label(t_img))
    [[2 0 0]
     [0 1 1]
     [0 0 1]]
    """
    lab_img, _ = label(in_img, **kwargs)
    new_lab_img = np.zeros_like(lab_img)
    lab_counts = [(i, np.sum(lab_img == i)) for i in
                  range(1, int(lab_img.max()) + 1)]
    lab_counts = sorted(lab_counts, key=lambda x: -x[1])
    for i, (j, _) in enumerate(lab_counts, 1):
        new_lab_img[lab_img == j] = i
    return new_lab_img


def reconstruct_from_patches(patches, image_size):
    """Reconstruct an nd image from all of its nd patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, ...)
        The complete set of patches. If the patches contain colour information
    image_size : tuple of ints
        the size of the image that will be reconstructed, the image dimension needs to be
        exactly one less than the patches dimensions
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image

    >>> import numpy as np
    >>> raw_img_2d = np.arange(24).reshape((4,6))
    >>> patches = extract_patches(raw_img_2d, (2, 3), 1).reshape((-1, 2, 3))
    >>> out_img = reconstruct_from_patches(patches, (4,6))
    >>> out_img.shape
    (4, 6)
    >>> int(np.sum(np.abs(out_img - raw_img_2d))*1000)
    0
    >>> raw_img_5d = np.arange(23040).reshape((4,6,8,10,12))
    >>> patches = extract_patches(raw_img_5d, (2, 3, 4, 5, 6), 1)
    >>> patches = patches.reshape((-1, 2, 3, 4, 5, 6))
    >>> out_img = reconstruct_from_patches(patches, (4,6,8,10,12))
    >>> out_img.shape
    (4, 6, 8, 10, 12)
    >>> int(np.sum(np.abs(out_img - raw_img_5d))*1000)
    0
    >>> reconstruct_from_patches_3d(np.zeros((32, 4, 4)), (8, 8))
    Traceback (most recent call last):
     ...
    AssertionError: Image size must be a 3- or 4-dimensional: Image Size:(8, 8)
    >>> reconstruct_from_patches_3d(np.zeros((32, 4, 4, 4)), (8, 8, 8, 3))
    Traceback (most recent call last):
      ...
    AssertionError: Patch dimensions should be one larger than image, Patch:(32, 4, 4, 4) and Image:(8, 8, 8, 3)
    """
    assert len(patches.shape) == len(
        image_size) + 1, "Patch dims should be one larger than image, " \
                         "Patch:{} and Image:{}".format(
        patches.shape, image_size)
    ip_dims = list(zip(image_size, patches.shape[1:]))
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_dims = [i_h - p_h + 1 for i_h, p_h in ip_dims]
    for p, ij_vec in zip(patches, product(*[range(n_h) for n_h in n_dims])):
        img[[slice(i, i + p_h) for i, (_, p_h) in zip(ij_vec, ip_dims)]] += p

    for ij_vec in product(*[range(i_h) for i_h, p_h in ip_dims]):
        # divide by the amount of overlap
        # XXX: is this the most efficient way? memory-wise yes, cpu wise?
        scale_f = float(np.prod([min(i + 1, p_h, i_h - i) for i, (i_h, p_h) in
                                 zip(ij_vec, ip_dims)]))

        img[ij_vec] /= scale_f

    return img


def reconstruct_from_patches_3d(patches, image_size):
    """Reconstruct a 3D image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width, patch_depth) or
        (n_patches, patch_height, patch_width, patch_depth, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_size : tuple of ints (image_height, image_width, image_depth) or
        (image_height, image_width, image_depth, n_channels)
        the size of the image that will be reconstructed
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image

    Examples
    --------

    >>> import numpy as np
    >>> raw_img = np.arange(64).reshape((4,4,4))
    >>> patches = extract_patches(raw_img, (2, 2, 2), 1).reshape((-1, 2, 2, 2))
    >>> out_img = reconstruct_from_patches_3d(patches, (4,4,4))
    >>> out_img.shape
    (4, 4, 4)
    >>> int(np.sum(np.abs(out_img - raw_img))*1000)
    0
    >>> ri_col = np.arange(192).reshape((4,4,4,3))
    >>> patches = extract_patches(ri_col, (2, 2, 2, 3), 1)
    >>> patches = patches.reshape((-1, 2, 2, 2, 3))
    >>> out_img = reconstruct_from_patches_3d(patches, (4, 4, 4, 3))
    >>> out_img.shape
    (4, 4, 4, 3)
    >>> int(np.sum(np.abs(out_img - ri_col))*1000)
    0
    >>> reconstruct_from_patches_3d(np.zeros((32, 4, 4)), (8, 8))
    Traceback (most recent call last):
     ...
    AssertionError: Image size must be a 3- or 4-dimensional: Image Size:(8, 8)
    >>> reconstruct_from_patches_3d(np.zeros((32, 4, 4, 4)), (8, 8, 8, 3))
    Traceback (most recent call last):
      ...
    AssertionError: Patch dimensions should be one larger than image, Patch:(32, 4, 4, 4) and Image:(8, 8, 8, 3)
    """
    assert len(image_size) == 3 or len(
        image_size) == 4, "Image size must be a 3- or 4-dimensional: Image Size:{}".format(
        image_size)
    assert len(patches.shape) == len(
        image_size) + 1, "Patch dimensions should be one larger than image, " \
                         "Patch:{} and Image:{}".format(
        patches.shape,
        image_size)
    i_h, i_w, i_d = image_size[:3]
    p_h, p_w, p_d = patches.shape[1:4]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_d = i_d - p_d + 1
    for p, (i, j, k) in zip(patches,
                            product(range(n_h), range(n_w), range(n_d))):
        img[i:i + p_h, j:j + p_w, k:k + p_d] += p

    for i in range(i_h):
        for j in range(i_w):
            for k in range(i_d):
                # divide by the amount of overlap
                # XXX: is this the most efficient way?
                # memory-wise yes, cpu wise?
                img[i, j, k] /= float(min(i + 1, p_h, i_h - i) *
                                      min(j + 1, p_w, i_w - j) *
                                      min(k + 1, p_d, i_d - k))
    return img


if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.nd import features

    doctest.testmod(features, verbose=True, optionflags=doctest.ELLIPSIS)
