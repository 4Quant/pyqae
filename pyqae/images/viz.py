"""
A little collection of visualization tools
"""

from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
import numpy as np

def show_image(test_img, cmap = 'bone', interp = 'none', **kwargs):
    fig, ax1 = plt.subplots(1,1)

    _ = ax1.imshow(test_img, cmap = cmap, interpolation =  interp, **kwargs)
    ax1.axis('off')
    return fig

def show_multi_images(*in_img):
    in_img = in_img[0] if len(in_img)==1 else in_img
    img_cnt = len(in_img)
    tile_w = np.ceil(np.sqrt(img_cnt)).astype(int)
    tile_h = np.ceil(img_cnt*1.0 / tile_w).astype(int)
    fig, maxs = plt.subplots(tile_w, tile_h, figsize = (10,10))
    fig.set_dpi(144)
    for cur_ax, cur_img in zip(maxs.flatten(), in_img):
        _ = cur_ax.imshow(cur_img, cmap = 'bone')
        cur_ax.axis('off')
    return fig

def show_image_slices(test_img, cmap = 'bone'):
    mont_img = montage2d(test_img)
    return show_image(mont_img, cmap = cmap)