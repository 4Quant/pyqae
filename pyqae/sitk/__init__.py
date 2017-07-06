"""
Tools for using SITK inside of PySpark
"""
import os
import warnings
from pyqae.utils import Dict, Any, List, Tuple, Union

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

try:
    from pyspark.sql import Row
    from pyspark.rdd import RDD
except ImportError:
    warnings.warn("Pyspark is not available using simplespark backend instead", ImportWarning)
    from pyqae.simplespark import Row
    from pyqae.simplespark import LocalRDD as RDD

_tag_lookup_dict = {}  # type: Dict[str, int]

try:
    from dicom._dicom_dict import DicomDictionary as __tag_dict

    _tag_lookup_dict = __tag_dict
except ImportError:
    warnings.warn("Pydicom library is not available header conversion is will not be available", ImportWarning)
    __tag_dict = {}  # just to make pycharm happy


def _try_to_hex(ckey):
    # type: (str) -> int
    try:
        return int('0x{}{}'.format(*ckey.split('|')), 0)
    except:
        return -1


def tag_lookup(ckey):
    # type: (str) -> str
    """
    A function for looking up tag names from the SimpleITK metadata tagging keys

    >>> assert tag_lookup('0008|0005') == 'SpecificCharacterSet', "Cant find simple code"
    >>> assert tag_lookup('bob') == 'bob', "Bob is simply bob"
    """
    return _tag_lookup_dict.get(_try_to_hex(ckey), [0, 0, 0, 0, ckey])[4]


class ITKImage(object):
    """
    A nicer, pythonic wrapper for the ITK data which can be serialized by spark and used for DataFrames easily
    """

    def __init__(self,
                 metadata,  # type: Dict[str, str]
                 itkdata,  # type: Dict[str, Any]
                 array,  # type: np.ndarray
                 verbose=False  # type: bool
                 ):
        """
        Create an ITK image from meta and array information
        :param metadata: Dict[(str, object)] the metadata information put in the Metadata dict
        :param itkdata: Dict[(str, object)] the items in the ITK image which can be set by a Set* command
        :param array: ndarray the array containing the data
        """
        assert isinstance(metadata, dict), "Metadata should be a dictionary, not {}".format(type(metadata))
        assert isinstance(itkdata, dict), "ITK Data should be a dictionary, not {}".format(type(itkdata))
        assert isinstance(array, np.ndarray), "Array should be a NumpyArray, not {}".format(type(array))
        self.metadata = metadata
        self.itkdata = itkdata
        self.array = array
        self.verbose = verbose

    def _create_empty_image(self):
        return sitk.Image(self.itkdata['itk_Size'],
                          self.itkdata['itk_PixelID'],
                          self.itkdata['itk_NumberOfComponentsPerPixel'])

    @property
    def dicomdata(self):
        """
        Dictionary with the keys replaced with DICOM data
        :return: Dict[(str,_)]
        """
        return {tag_lookup(ckey): cval for ckey, cval in self.metadata.copy().items()}

    def create_image(self):
        """
        Create a SimpleITK image from the class
        :return: a standard SimpleITK image
        """
        res_img = sitk.GetImageFromArray(self.array)
        for key, new_val in self.itkdata.items():
            try:
                getattr(res_img, key.replace("itk_", "Set"))(new_val)
                if self.verbose:
                    print("Set {} to {}".format(key, new_val))
            except:
                if self.verbose:
                    print("Can't set {} to {}".format(key, new_val))
        for key, val in self.metadata.items():
            res_img.SetMetaData(key, val)
        return res_img

    def _asdict(self):
        """
        Converts the object into a dictionary ready for use as a DataFrame
        :return:
        """
        itkdict = dict(list(self.metadata.items()) + list(self.itkdata.items()))
        itkdict['array'] = self.array.tolist()  # make it serializable
        itkdict['array_dtype'] = str(self.array.dtype)
        return itkdict

    def show(self):
        show_itk_image(self.create_image())

    def save(self, path, useCompression=False):
        """
        Save the image to an output path
        :param path: str the path to save to
        :param useCompression: bool whether or not to compress (not supported with all file formats
        :return:
        """
        return sitk.WriteImage(self.create_image(), path, useCompression)

    def __repr__(self):
        return "<{cname} Size:{itk_Size} Type:{itk_PixelIDTypeAsString}>".format(cname=self.__class__.__name__,
                                                                                 **self.itkdata)

    @staticmethod
    @property
    def _get_validate_fields():
        # type: () -> List[str]
        return """array, array_dtype, itk_Depth, itk_Dimension,
        itk_Direction, itk_Height, itk_NumberOfComponentsPerPixel,
        itk_Origin, itk_PixelID, itk_PixelIDTypeAsString,
        itk_PixelIDValue, itk_Size, itk_Spacing, itk_Width""".replace("\n", "").replace(" ", "").split(",")

    @staticmethod
    def _has_fields(in_fields):

        field_to_check = ITKImage._get_validate_fields
        for cfield in field_to_check:
            assert cfield in in_fields, "DataFrame is missing column:{}, {}".format(cfield, in_fields)

    @staticmethod
    def _parse_dict(in_dict):
        ITKImage._has_fields(in_dict.keys())
        arr = np.array(in_dict.pop('array')).astype(in_dict.pop('array_dtype'))
        itkdata = {}
        metadata = {}
        for k, v in in_dict.items():
            if k.startswith('itk_'):
                itkdata[k] = v
            else:
                metadata[k] = "{}".format(v)
        return ITKImage(metadata, itkdata, arr)

    @staticmethod
    def load_file(img_path):
        """
        Reads an ITK image to a serializable python tuple with the fields extracted
        :param img_path:
        :return:
        """

        itk_img = sitk.ReadImage(img_path)
        return ITKImage.convert(itk_img)

    @staticmethod
    def convert(itk_img):
        itk_keys = ['Depth',
                    'Dimension',
                    'Direction',
                    'Height',
                    'NumberOfComponentsPerPixel',
                    'Origin',
                    'PixelID',
                    'PixelIDTypeAsString',
                    'PixelIDValue',
                    'Size',
                    'Spacing',
                    'Width']
        metadata = dict(map(lambda x: (x, itk_img.GetMetaData(x)), itk_img.GetMetaDataKeys()))
        itkdata = dict([("itk_{}".format(key), getattr(itk_img, "Get{}".format(key))()) for key in itk_keys])
        return ITKImage(metadata, itkdata, sitk.GetArrayFromImage(itk_img))


def apply_filter(in_df, filter_op):
    """
    Apply a SimpleITK filter to a given image
    :param in_df: DataFrame made from a series of ITK files
    :param filter_op: sitk.Image => sitk.Image an operation which converts one image to another
    :return: DataFrame containing the new image data
    """
    itk_rdd = in_df.rdd.map(lambda x: ITKImage._parse_dict(x.asDict()))
    post_rdd = itk_rdd.map(lambda x: Row(**ITKImage.convert(filter_op(x.create_image()))._asdict()))
    return post_rdd.toDF()


def show_first_image(in_df):
    """
    Show the first image in the result
    :param in_df:
    :return:
    """
    return in_df.rdd.map(lambda x: ITKImage._parse_dict(x.asDict())).first().show()


def fix_df_col_names(t_prev_df):
    """
    Since the column names need to be valid (see banned characters) in order to be saved as parquet
    :param t_prev_df: DataFrame to fix
    :return: DataFrame with renamed columns
    """
    new_df = t_prev_df
    for col in t_prev_df.columns:
        new_col = col
        for fix_chr in ' ,;{}()\n\t=':
            new_col = new_col.replace(fix_chr, "")
        new_df = new_df.withColumnRenamed(col, new_col)
    return new_df


def save_itk_local(in_df, base_path, allow_overwrite=False, file_ext='tif', useCompression=False):
    """
    Save an ITK dataframe on local (shared directory) paths
    :param in_df: DataFrame of the itk images
    :param base_path: string the directory to save in
    :param allow_overwrite: bool allow overwrite to occur
    :param file_ext: string the extension to save to
    :param useCompression: bool compress the output
    """
    try:
        os.mkdir(base_path)
    except:
        print('{} already exists'.format(base_path))
        if not allow_overwrite: raise RuntimeError(
            "Overwriting has not been enabled! Please remove directory {}".format(base_path))

    def _write(x):
        o_row, uid = x
        new_path = os.path.join(base_path, "{}.{}".format(uid, file_ext))
        ITKImage._parse_dict(o_row.asDict()).save(new_path, useCompression=useCompression)
        return new_path

    return base_path, in_df.rdd.zipWithUniqueId().map(_write).cache().collect()


def itk_obj_to_pandas(obj_lst):
    import pandas as pd
    return pd.DataFrame([in_obj._asdict() for in_obj in obj_lst])


def _read_to_rdd(paths, context=None, **kwargs):
    """
    Read a list of paths, list of lists, or RDD of paths to a RDD of ITK dictionaries
    :param paths:
    :param sc: SparkContext only needed if paths is not an RDD
    :param kwargs: additional arguments for creating the RDD with parallelize
    :return:
    """
    if not isinstance(paths, RDD):
        assert context is not None, "Cannot ignore sc argument if paths is not an RDD, paths:{}".format(paths)
        path_rdd = context.parallelize(paths, **kwargs)
    else:
        path_rdd = paths
    return path_rdd.map(lambda cpath: (cpath, ITKImage.load_file(cpath)._asdict()))


def read_to_dataframe(paths, context=None, **kwargs):
    """
    Create a dataframe from a list of paths, list of lists, or RDD of paths
    :param paths: files to load to make the ITK DataFrame
    :param context: SparkContext (only needed if paths is not an RDD)
    :param kwargs: additional arguments for creating the RDD with parallelize
    :return: DataFrame with all relevant fields see ITKImage
    """
    cur_rdd = _read_to_rdd(paths, context=context, **kwargs)
    return cur_rdd.map(lambda x: Row(filename=x[0], **x[1])).toDF()


def show_itk_image(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if (slicer):
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    def callback(z=None):

        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        plt.set_cmap("gray")

        if z is None:
            ax.imshow(nda, extent=extent, interpolation=None)
        else:
            ax.imshow(nda[z, ...], extent=extent, interpolation=None)

        if title:
            plt.title(title)

        plt.show()
        return fig

    try:
        from ipywidgets import interact
        return interact(lambda z: callback(z), z=(0, nda.shape[0] - 1))
    except ImportError:
        if slicer: print("Slicer can only be used inside of IPython")
        slicer = False
        interact = None
        return callback()



def show_itk_image3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)
    return show_itk_image(img, title, margin, dpi)


def _get_spacing_np(in_data):
    # type: (sitk.Image) -> np.ndarray
    """
    the dimensions are switched in the numpy and sitk-based representations so we manually correct it here
    :param in_data:
    :return:
    """
    gs = tuple(in_data.GetSpacing())
    if len(gs) == 3:
        gs = gs[2], gs[0], gs[1]
    elif len(gs) == 2:
        gs = gs[1], gs[0]
    return np.array(gs)


from scipy.ndimage import zoom


def extract_custom_array(in_data,  # type: sitk.Image
                         new_vox_size,  # type: Union[float, np.ndarray]
                         order=0,
                         verbose=False,
                         **kwargs):
    # type: (...) -> Tuple[np.array, List[float]]
    """
    Forces the 3d array returned to be isotropic so the rendering looks correct
    :param in_data: the sitk image to extract
    :param new_vox_size the new voxel size
    :param order: the interpolation to use (0-> nearest neighbor, 2 -> cubic)
    :param verbose: print out information on sizing and changes
    :param kwargs:
    :return:
    """

    gs_arr = _get_spacing_np(in_data).tolist()
    t_array = sitk.GetArrayFromImage(in_data)
    new_v_size = new_vox_size if isinstance(new_vox_size, np.ndarray) else np.array([new_vox_size] * len(gs_arr))
    assert len(new_v_size) == len(gs_arr), "Voxel size and old spacing must have the same size {}, {}".format(
        new_v_size, gs_arr)
    scale_f = np.array(gs_arr) / new_v_size
    if verbose: print(gs_arr, '->', new_v_size, ':', scale_f)
    return zoom(t_array, scale_f, order=order, **kwargs), list(new_v_size.tolist())


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
