"""
A little collection of visualization tools
"""

from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
import numpy as np
import warnings

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


def show_lazy_browser(img, tile_size = (256, 256), title=None, margin=0.05, dpi=80, cmap = "gray" ):
    """
    Show a browser for a lazy image which automatically loads / calculates the tiles as they are needed
    :param img: a DiskMappedLazyImage or BoltArray for viewing interactively
    :param tile_size: the size of the tile to show (smaller is faster)
    :param title: a label if desired
    :param margin: boundaries around axes
    :param dpi: the dpi for the figure
    :param cmap: the colormap to use
    :return: an interactive viewer (if inside ipython)
    """
    slicer = True
    def callback(x=0, y=0):
        fig = plt.figure(dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

        plt.set_cmap(cmap)
        cur_img = img[x:x+tile_size[0], y:y+tile_size[1]]
        cur_img = cur_img if isinstance(cur_img, np.ndarray) else cur_img.toarray()
        ax.imshow(cur_img,interpolation=None)

        if title:
            plt.title(title)

        return fig

    if slicer:
        try:
            from ipywidgets import interact, interactive
            from ipywidgets import widgets

            interact(lambda x, y: callback(x, y), x=(0, img.shape[0] - tile_size[0]),
                     y=(0, img.shape[1] - tile_size[1]),
                     continuous_update=False)
        except ImportError:
            warnings.warn("Interactive lazy viewer can only be used inside of IPython", ImportWarning)
            return callback()
    else:
        return callback()