from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra / matrices
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x


def comp_images(old_ds, new_ds, z_slice=0):
    batch_idx = np.array(range(old_ds.shape[0]))
    np.random.shuffle(batch_idx)
    fig, ax_all = plt.subplots(2, 2, figsize=(8, 8))
    for pid, (c_raw_ax, c_flt_ax) in zip(batch_idx, ax_all.T):
        c_raw_ax.imshow(old_ds[pid, z_slice, :, :].clip(0, 1), cmap='gray', vmin=0, vmax=1)
        c_raw_ax.set_title("Raw Image:{}".format(pid))
        c_flt_ax.imshow(new_ds[pid, z_slice, :, :].clip(0, 1), cmap='gray', vmin=0, vmax=1)
        c_flt_ax.set_title("Transformed Image:{}".format(pid))


def preprocess(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[3]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=0.1, multichannel=False)
    return X


def full_augmentation(X, Y=None, scale_range=1, angle_range=0, h_range=0, w_range=0,
                      flip_lr=True, flip_ud=True,
                      add_noise_level=0, mul_noise_level=0,
                      proportional=False, volume_preserving=False,
                      Y_noise=False, boundary_mode='nearest'):
    """
    The full suite of standard augmentation functions with jitter, rotation, flipping, scaling, and noise
    """
    if proportional: assert volume_preserving == False, "Cannot be both proportional and volume preserving"
    X_scale = np.copy(X)
    if Y is not None:
        Y_scale = np.copy(Y)
    size = X.shape[2:]
    trn_mat = np.eye(2).astype(np.float32)
    aff_func = lambda c_vol, c_mat: ndimage.affine_transform(c_vol, c_mat, mode=boundary_mode, output_shape=c_vol.shape)
    rot_func = lambda c_vol, c_angle: ndimage.rotate(c_vol, c_angle, reshape=False, order=1, mode=boundary_mode)
    shift_func = lambda c_vol, c_vec: ndimage.shift(c_vol, c_vec, order=0, mode=boundary_mode)
    for i in range(len(X)):
        scale_x = (100.0 + np.random.randint(-scale_range, scale_range)) / 100.0
        if not proportional:
            scale_y = (100.0 + np.random.randint(-scale_range, scale_range)) / 100.0
        elif volume_preserving:
            scale_y = 1 / scale_x
        else:
            scale_y = scale_x
        trn_mat[0, 0] = scale_x
        trn_mat[1, 1] = scale_y
        if angle_range > 0:
            angle = np.random.randint(-angle_range, angle_range)
        else:
            angle = 0

        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])

        do_flip_lr = np.random.randint(0, 2) & flip_lr
        do_flip_ud = np.random.randint(0, 2) & flip_ud

        for j in range(X.shape[1]):
            X_scale[i, j] = aff_func(X[i, j], trn_mat)
            X_scale[i, j] = rot_func(X_scale[i, j], angle)
            X_scale[i, j] = shift_func(X_scale[i, j], (h_shift, w_shift))
            if do_flip_lr: X_scale[i, j] = np.fliplr(X_scale[i, j])
            if do_flip_ud: X_scale[i, j] = np.flipud(X_scale[i, j])
            mul_noise = (100.0 + 2 * mul_noise_level * (np.random.random(X.shape[2:4]) - 0.5)) / 100.0
            add_noise = add_noise_level * (np.random.random(X.shape[2:4]))
            X_scale[i, j] = (X_scale[i, j] * mul_noise + add_noise).astype(X_scale.dtype)
        if Y is not None:
            for j in range(Y.shape[1]):
                Y_scale[i, j] = aff_func(Y[i, j], trn_mat)
                Y_scale[i, j] = rot_func(Y_scale[i, j], angle)
                Y_scale[i, j] = shift_func(Y_scale[i, j], (h_shift, w_shift))
                if do_flip_lr: Y_scale[i, j] = np.fliplr(Y_scale[i, j])
                if do_flip_ud: Y_scale[i, j] = np.flipud(Y_scale[i, j])
                if Y_noise:
                    mul_noise = (100.0 + 2 * mul_noise_level * (np.random.random(Y.shape[2:4]) - 0.5)) / 100.0
                    add_noise = add_noise_level * (np.random.random(Y.shape[2:4]))
                    Y_scale[i, j] = (Y_scale[i, j] * mul_noise + add_noise).astype(Y_scale.dtype)
    if Y is not None:
        return X_scale, Y_scale
    else:
        return X_scale


def stack_augmentation(train_img, lab_img, stack_count=5, *args, **kwargs):
    """
    Run augumentation the data set several times and append the results together into one big output
    """
    train_list = []
    lab_list = []
    for i in range(stack_count):
        i_train, i_labs = full_augmentation(train_img, lab_img, *args, **kwargs)
        train_list += [i_train]
        lab_list += [i_labs]

    return np.vstack(train_list), np.vstack(lab_list)


def add_grid(t_img, max_x=1, max_y=1):
    """
    Add a position grid in x and y to the image so the region can be identified
    """
    xx, yy = np.meshgrid(np.linspace(-max_y, max_y, t_img.shape[3]),
                         np.linspace(-max_x, max_x, t_img.shape[2]))
    out_arr = []
    for it_img in t_img:
        out_arr += [np.stack(it_img.tolist() + [xx, yy])]
    return np.stack(out_arr)


def tile_extract(t_img, n_img=10, d_x=128, d_y=128):
    imgs, channels, xwid, ywid = t_img.shape
    out_arr = []
    for i in range(n_img):
        img_idx = np.random.randint(imgs)
        xs = np.random.randint(xwid - d_x)
        ys = np.random.randint(ywid - d_y)
        c_chunk = t_img[img_idx, :, xs:(xs + d_x), ys:(ys + d_y)]
        out_arr += [c_chunk]
    return np.stack(out_arr)


def dual_tile_extract(t_img, l_img, n_img=10, d_x=128, d_y=128):
    """
    Extracts matching tiles from two images (training and labels) 
    """
    assert len(t_img.shape) == 4, "Input image must be 4 dimensional tensor : {}".format(t_img.shape)
    assert len(l_img.shape) == len(t_img.shape), "Input and label image should have same dimensions {}:{}".format(
        l_img.shape, t_img.shape
    )
    imgs, channels, xwid, ywid = t_img.shape
    out_arr, out_lab = [], []
    for i in range(n_img):
        img_idx = np.random.randint(imgs)
        xs = np.random.randint(xwid - d_x)
        ys = np.random.randint(ywid - d_y)
        out_arr += [t_img[img_idx, :, xs:(xs + d_x), ys:(ys + d_y)]]
        out_lab += [l_img[img_idx, :, xs:(xs + d_x), ys:(ys + d_y)]]
    return np.stack(out_arr), np.stack(out_lab)


from pyqae.nd import uniform_nd_bias_sampler
from scipy.ndimage.filters import maximum_filter


def dual_tile_sample(t_img, l_img,
                     n_img=10, n_tiles=5,
                     d_x=128, d_y=128,
                     base_p=0.5, cutoff_val=0.25,
                     sample_slices=False):
    """

    :param t_img:
    :param l_img:
    :param n_img: number of distinct images to take
    :param n_tiles: number of tiles per image (more efficient than single tiles)
    :param d_x:
    :param d_y:
    :param base_p:
    :param cutoff_val:
    :param sample_slices:
    :return:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> t_img = np.arange(100).reshape((1,1,10,10))
    >>> dual_tile_sample(t_img, t_img>50, 1, 2, d_x = 5, d_y = 5)[0]
    array([[[[33, 34, 35, 36, 37],
             [43, 44, 45, 46, 47],
             [53, 54, 55, 56, 57],
             [63, 64, 65, 66, 67],
             [73, 74, 75, 76, 77]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[42, 43, 44, 45, 46],
             [52, 53, 54, 55, 56],
             [62, 63, 64, 65, 66],
             [72, 73, 74, 75, 76],
             [82, 83, 84, 85, 86]]]])
    >>> dual_tile_sample(t_img, np.abs(t_img-50)<10, 1, 2, d_x = 2, d_y = 2)[0]
    array([[[[43, 44],
             [53, 54]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[57, 58],
             [67, 68]]]])
    """
    assert len(t_img.shape) == 4, "Input image must be 4 dimensional tensor : {}".format(t_img.shape)
    assert len(l_img.shape) == len(t_img.shape), "Input and label image should have same dimensions {}:{}".format(
        l_img.shape, t_img.shape
    )
    imgs, channels, xwid, ywid = t_img.shape
    out_arr, out_lab = [], []
    if sample_slices:
        img_slices = uniform_nd_bias_sampler(l_img, count=n_img, base_p=base_p, axis=0, cut_off_val=cutoff_val)
    else:
        img_slices = np.random.randint(imgs, size=(n_img,))

    for img_idx in img_slices:
        # since we only look between 0 and xwid-d_x
        e_prob_tile = maximum_filter(l_img[img_idx, :], (1, d_x, d_y))[:, :(xwid - d_x), :(ywid - d_y)]
        xs_pts = uniform_nd_bias_sampler(e_prob_tile, count=n_tiles, base_p=base_p, axis=1,
                                    cut_off_val=cutoff_val, ignore_zeros=True)
        ys_pts = uniform_nd_bias_sampler(e_prob_tile, count=n_tiles, base_p=base_p, axis=2,
                                    cut_off_val=cutoff_val, ignore_zeros=True)
        for (xs, ys) in zip(xs_pts, ys_pts):
            out_arr += [t_img[img_idx, :, xs:(xs + d_x), ys:(ys + d_y)]]
            out_lab += [l_img[img_idx, :, xs:(xs + d_x), ys:(ys + d_y)]]
    return np.stack(out_arr), np.stack(out_lab)


def cache_generator(i_gen, loops):
    """
    Extracts a given number of loops from the generator 
    and packages them as a standard dataset. This can make multiple epoch 
    training much faster when training on cpu
    """
    train_ds = [(X, y) for i, (X, y) in zip(range(loops), i_gen)]
    return np.vstack([X for (X, y) in train_ds]), np.vstack([y for (X, y) in train_ds])


if __name__ == "__main__":
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae.dnn import augmentation

    doctest.testmod(augmentation, verbose=True, report=True)
