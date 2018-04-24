import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image as PImage


def pull_img_http(base_url):
    response = requests.get(base_url)
    if response.ok:
        return PImage.open(BytesIO(response.content))
    return None


def pull_img_http_array(base_url, out_size=None):
    out_img = pull_img_http(base_url)
    if out_img is None: return out_img
    if out_size is not None: out_img = out_img.resize(out_size, PImage.BICUBIC)
    return np.array(out_img)


def pull_img_http_path(out_folder, url, uid, row, **kwargs):
    s_path = os.path.join(out_folder, '%05d-%s.png' % (row, uid))
    if not os.path.exists(s_path):
        c_img = pull_img_http(url)
        c_img.save(s_path)
    return s_path
