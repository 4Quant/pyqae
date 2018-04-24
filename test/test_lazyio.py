import pytest
import os
resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')

def test_backends():
    from pyqae.images.lazy import backends, DiskMappedLazyImage
    # check boundary issues for tiles
    img_path = os.path.join(resources, "singlelayer_tif", "dot1_grey_lzw.tif")
    for c_back in backends:
        c_image = DiskMappedLazyImage(img_path, c_back)
        for i in [0, c_image.shape[0]-20, c_image.shape[0]]:
            for j in [0, c_image.shape[1]-20, c_image.shape[1]]:
                try:
                    o_shape = c_image[i:i+20, j:j+20].shape
                except:
                    o_shape = 'Failed'
                print(c_back, (i,j), o_shape)