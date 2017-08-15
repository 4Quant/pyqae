import numpy as np

from pyqae.nd import meshgridnd_like
from pyqae.utils import pprint, \
    get_error  # noinspection PyUnresolvedReferences

"""
The segmentation tools which are commonly used for a large number of different applications and thus part of the core PYQAE framework

"""
from skimage.morphology import disk, ball, dilation
from scipy.ndimage import label


def hysteresis_labeling(in_img, label_cutoff,
                        fill_cutoff,
                        max_iters=-1):
    """
    Run hysteresis thresholding on an image with a lower and upper cutoff
    :param in_img:
    :param low_cutoff: the cutoff for identifying the original cores
    :param higher_cutoff: the cutoff
    :param max_iters: maximum number of iterations to run
    :return:
    >>> t_mat = np.arange(0,9).reshape((3,3))+1
    >>> t_mat += 10*(t_mat % 4 == 1)
    >>> pprint(t_mat)
    [[11  2  3]
     [ 4 15  6]
     [ 7  8 19]]
    >>> pprint(hysteresis_labeling(t_mat, 10, 0))
    [[1 2 3]
     [2 2 3]
     [3 3 3]]
    """
    lab_img, _ = label(in_img > label_cutoff)
    mask_img = in_img > fill_cutoff
    return hysteresis_growing(lab_img, mask_img, max_iters=max_iters)


def show_hysteresis_labeling(in_img,
                             label_cutoff,
                             fill_cutoff,
                             max_iters=-1,
                             slice_idx=None):
    """
    Show a figure with the labeling
    :param in_img:
    :param label_cutoff:
    :param fill_cutoff:
    :param max_iters:
    :return:
    """
    import matplotlib.pyplot as plt
    if len(in_img.shape) == 3:
        if slice_idx == None:
            slice_idx = in_img.shape[0] // 2
        show_fcn = lambda x: x[slice_idx]
    elif len(in_img.shape) == 2:
        show_fcn = lambda x: x
    else:
        raise NotImplementedError('ND arrays have not been implemented only '
                                  '2 and 3D - {}'.format(in_img.shape))
    fig, n_axs = plt.subplots(1, 4, figsize=(16, 4))
    [c_ax.axis('off') for c_ax in n_axs]
    (ax1, ax2, ax3, ax4) = n_axs
    ax1.imshow(show_fcn(in_img))

    ax2.imshow(show_fcn(in_img > label_cutoff))
    ax2.set_title('Core Label Points')
    ax3.imshow(show_fcn(in_img > fill_cutoff))
    ax3.set_title('Filling Region')
    ax4.imshow(show_fcn(hysteresis_labeling(in_img,
                                            label_cutoff=label_cutoff,
                                            fill_cutoff=fill_cutoff,
                                            max_iters=max_iters)),
               cmap='gist_earth')
    ax4.set_title('Hysteresis Labeled Image')
    return fig


def hysteresis_growing(in_labels, in_mask, max_iters=-1):
    """
    Grow labels into a mask
    :param in_labels:
    :param in_mask:
    :param max_iters: maximum number of iterations to run
    :return:
    >>> t_lab = np.arange(0,9).reshape((3,3))+1
    >>> t_lab *= (t_lab % 4 == 1)
    >>> pprint(t_lab)
    [[1 0 0]
     [0 5 0]
     [0 0 9]]
    >>> pprint(hysteresis_growing(t_lab, np.ones_like(t_lab), max_iters = 1))
    [[1 5 0]
     [5 5 9]
     [0 9 9]]
    >>> pprint(hysteresis_growing(t_lab, np.ones_like(t_lab)))
    [[1 5 9]
     [5 5 9]
     [9 9 9]]
    """
    if in_labels.shape != in_mask.shape:
        raise ValueError(
            'Label and mask dimensions must match {} != {}'.format(
                in_labels.shape, in_mask.shape))
    if len(in_labels.shape) == 2:
        s_elem = disk(np.sqrt(2))
    elif len(in_labels.shape) == 3:
        s_elem = ball(np.sqrt(3))
    else:
        raise ValueError('Only 2- and 3D images are supported: {}'.format(
            in_labels.shape))
    still_changing = True

    old_labels = in_labels.copy()
    iters = 1
    while still_changing:
        frozen_pixels = (old_labels > 0)
        new_labels = dilation(old_labels, s_elem)
        # restore frozen pixels
        new_labels[frozen_pixels] = old_labels[frozen_pixels]
        # zero points outside the mask
        new_labels *= in_mask
        if np.sum(new_labels - old_labels) < 1:
            still_changing = False
        elif (iters > max_iters) and (max_iters > 0):
            still_changing = False
        else:
            old_labels = new_labels
        iters += 1
    return old_labels


def remove_edges(in_img,
                 radius,
                 sph_mode=True):
    # type: (np.ndarray, float, bool) -> np.ndarray
    """
    The function removes all values (with a simple mask) outside of the radius

    :param in_img: the image to process
    :param radius: the radius inside of which values are kept
    :param sph_mode: use spherical coordinates otherwise cylindrical are used
    :return:
    >>> remove_edges(np.ones((4,4,4)), 1.0, True).astype(np.int8)
    array([[[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]], dtype=int8)
    >>> remove_edges(np.ones((3,3,3)), 0.5, False).astype(np.int8)[:,:,0]
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]], dtype=int8)
    >>> remove_edges(np.ones((3,3,3)), 0.5, False).astype(np.int8)
    array([[[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]], dtype=int8)

    """
    xx, yy, zz = meshgridnd_like(in_img,
                                 rng_func=lambda n: np.linspace(-1, 1, num=n))
    cdist = lambda x: np.power(
        (x.astype(np.float32) - x.mean()) / (x.max() - x.mean()), 2)
    cdist = lambda x: np.power(x, 2)
    dist_img = cdist(xx) + cdist(yy)
    if sph_mode:
        dist_img += cdist(zz)
    return in_img * (np.sqrt(dist_img) < radius)


def remove_edge_objects(in_img):
    # type: (np.ndarray) -> np.ndarray
    # just get the boundary labels
    edge_labels = np.unique(np.pad(in_img, 1, mode='edge') - np.pad(in_img, 1,
                                                                    mode=lambda
                                                                        *args: 0))
    all_labels = np.unique(in_img)
    clean_lung_comp = np.zeros_like(in_img)
    size_sorted_labels = sorted(
        filter(lambda x: x not in edge_labels, all_labels),
        key=lambda grp_label: -1 * np.sum(in_img == grp_label))
    for new_label, old_label in enumerate(size_sorted_labels):
        clean_lung_comp[in_img == old_label] = new_label + 1
    return clean_lung_comp


def remove_small_and_sort_labels(in_lab_img, min_label_count):
    """
    Remove labels with fewer than a given number of voxels
    :param in_lab_img:
    :param min_label_count:
    :return:

    >>> from skimage.measure import label
    >>> import numpy as np
    >>> remove_small_and_sort_labels(label(np.eye(3)),0)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    >>> np.random.seed(1234)
    >>> remove_small_and_sort_labels(np.random.randint(0, 4, size = (8,8)), 1)
    array([[1, 1, 2, 3, 0, 0, 0, 3],
           [1, 3, 1, 3, 2, 2, 1, 2],
           [0, 0, 2, 2, 2, 0, 0, 0],
           [3, 0, 3, 1, 2, 2, 1, 2],
           [0, 1, 0, 3, 2, 2, 2, 1],
           [1, 1, 0, 3, 1, 0, 1, 2],
           [1, 0, 3, 1, 1, 1, 2, 3],
           [2, 1, 1, 0, 2, 1, 2, 0]])
    """
    clean_lab_img = np.zeros_like(in_lab_img)
    size_sorted_labels = sorted(filter(
        lambda grp_label: np.sum(in_lab_img == grp_label) > min_label_count,
        np.unique(in_lab_img[in_lab_img > 0])),
        key=lambda grp_label: -1 * np.sum(
            in_lab_img == grp_label))
    for new_label, old_label in enumerate(size_sorted_labels):
        clean_lab_img[in_lab_img == old_label] = new_label + 1
    return clean_lab_img


from skimage.morphology import convex_hull as ch


def convex_hull_slice(in_img):
    # type: (np.ndarray) -> np.ndarray
    """
    A safe version of convex-hull that can be used on empty images
    :param in_img:
    :return:
    >>> pprint(convex_hull_slice(np.eye(3)))
    [[ True False False]
     [False  True False]
     [False False  True]]
    >>> pprint(convex_hull_slice(np.zeros((3,3))))
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]]
    >>> get_error(ch.convex_hull_image,image = np.zeros((3,3)))
    'No points given'
    """
    if in_img.max() == 0:
        return in_img
    try:
        return ch.convex_hull_image(in_img)
    except ValueError:
        # occurs when no points are found, just return the empty image
        return in_img


def convex_hull_3d(in_img):
    # type: (np.ndarray) -> np.ndarray
    """
    A slice-based 3D implementation of convex-hull
    :param in_img:
    :return:
    >>> t_image = np.stack([np.eye(3), 1-np.eye(3)],0).astype(int)
    >>> pprint(convex_hull_3d(t_image))
    [[[ True False False]
      [False  True False]
      [False False  True]]
    <BLANKLINE>
     [[False  True  True]
      [ True  True  True]
      [ True  True False]]]
    """
    return np.stack([convex_hull_slice(np.array(c_slice > 0)) for c_slice in
                     in_img])


from skimage.segmentation import clear_border


def clear_border_3d(in_labels):
    # type: (np.ndarray) -> np.ndarray
    """
    A 3d implementation of the clear border function
    :param in_labels: either a binary or segmented image
    :return: the image with clear border run on every slice
    >>> t_image = np.stack([np.eye(3)*(i+1) for i in range(3)],0).astype(int)
    >>> clear_border_3d(t_image).max()
    0
    >>> t_image[1,0,0] = 0
    >>> t_image[1,2,2] = 0
    >>> pprint(clear_border_3d(t_image))
    [[[0 0 0]
      [0 0 0]
      [0 0 0]]
    <BLANKLINE>
     [[0 0 0]
      [0 2 0]
      [0 0 0]]
    <BLANKLINE>
     [[0 0 0]
      [0 0 0]
      [0 0 0]]]
    """
    return np.stack([clear_border(c_seg) for c_seg in in_labels], 0)
