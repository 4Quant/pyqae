"""
Package for visualization tools and support
"""
from io import BytesIO
from skimage.io import imread
from PIL import Image as PImage
import numpy as np
import base64
from matplotlib.pyplot import cm


def _np_to_uri(in_array, # type: np.ndarray
               cmap='RdBu'):
    """
    Convert a numpy array to a data URI with a png inside

    >>> _np_to_uri(np.zeros((100,100)))
    'iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAABUElEQVR4nO3SQQEAEADAQBQRT/8ExPDYXYI9Ns/Yd5C1fgfwlwHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxDxypAoX8C2RlAAAAAElFTkSuQmCC'
    """
    test_img_data = np.array(in_array).astype(np.float32)
    test_img_data -= test_img_data.mean()
    test_img_data /= test_img_data.std()
    test_img_color = cm.get_cmap(cmap)((test_img_data + 0.5).clip(0, 1))
    test_img_color *= 255
    p_data = PImage.fromarray(test_img_color.clip(0, 255).astype(np.uint8))
    rs_p_data = p_data.resize((128, 128), resample=PImage.BICUBIC)
    out_img_data = BytesIO()
    rs_p_data.save(out_img_data, format='png')
    out_img_data.seek(0) # rewind
    return base64.b64encode(out_img_data.read()).decode("ascii").replace("\n", "")

_wrap_uri = lambda data_uri: "data:image/png;base64,{0}".format(data_uri)

def display_uri(uri_list, wrap_ipython = True):
    """

    show_uri(_np_to_uri(np.zeros((100,100))))

    """
    fake_HTML = lambda x: x  # dummy function
    try:
        from IPython.display import HTML
    except ImportError:
        wrap_ipython = False
        HTML = fake_HTML
    if not wrap_ipython: HTML = fake_HTML
    out_html = ""
    for in_uri in uri_list:
        out_html += """<img src="{0}" width = "100px" height = "100px" />""".format(_wrap_uri(in_uri))
    return HTML(out_html)


from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
MAX_COMP_LIMIT = 50


def draw_3d_labels(in_bone_labels, # type: np.ndarray
                  start_idx = 1):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    cmap = plt.cm.get_cmap('nipy_spectral_r')
    max_comp = in_bone_labels.max()

    ax_flip = lambda x: x[::-1,::-1].swapaxes(0,2).swapaxes(0,1)
    for i in range(start_idx,min(max_comp, MAX_COMP_LIMIT)):
        if i==0:
            v_img = ax_flip((in_bone_labels>0).astype(np.float32))
        else:
            v_img = ax_flip((in_bone_labels==i).astype(np.float32))

        verts, faces = marching_cubes(v_img, level = 0)
        mesh = Poly3DCollection(verts[faces])

        if i>0:
            mesh.set_facecolor(cmap(i/max_comp)[:3])
            mesh.set_alpha(0.75)
        else:
            mesh.set_facecolor([1,1,1])
            mesh.set_edgecolor([0,0,0])
            mesh.set_alpha(0.1)
        #mesh.set_edgecolor(cmap(i/max_comp)[:3])

        ax.add_collection3d(mesh)
    n_shape = ax_flip(in_bone_labels).shape
    ax.set_xlim(0, n_shape[0])
    ax.set_ylim(0, n_shape[1])
    ax.set_zlim(0, n_shape[2])
    ax.axis('off')
    return fig