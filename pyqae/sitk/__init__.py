"""
Tools for using SITK inside of PySpark
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

try:
    from pyspark.sql import Row
    from pyspark.rdd import RDD
except ImportError:
    print("Pyspark is not available using simplespark backend instead")
    from ..simplespark import Row
    from ..simplespark import LocalRDD as RDD

class ITKImage(object):
    """
    A nicer, pythonic wrapper for the ITK data which can be serialized by spark and used for DataFrames easily
    """
    def __init__(self, metadata, itkdata, array, verbose = False):
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
        return sitk.Image(self.itkdata['itk_Size'], self.itkdata['itk_PixelID'],
                          self.itkdata['itk_NumberOfComponentsPerPixel'])

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

    def save(self, path, useCompression = False):
        """
        Save the image to an output path
        :param path: str the path to save to
        :param useCompression: bool whether or not to compress (not supported with all file formats
        :return:
        """
        return sitk.WriteImage(self.create_image(), path, useCompression)
    @staticmethod
    def _parse_dict(in_dict):
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
    def read_itk_as_obj(img_path):
        """
        Reads an ITK image to a serializable python tuple with the fields extracted
        :param img_path:
        :return:
        """
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
        itk_img = sitk.ReadImage(img_path)
        metadata = dict(map(lambda x: (x, itk_img.GetMetaData(x)), itk_img.GetMetaDataKeys()))
        itkdata = dict([("itk_{}".format(key), getattr(itk_img, "Get{}".format(key))()) for key in itk_keys])
        return ITKImage(metadata, itkdata, sitk.GetArrayFromImage(itk_img))

def itk_obj_to_pandas(obj_lst):
    import pandas as pd
    return pd.DataFrame([in_obj._asdict() for in_obj in obj_lst])

def _read_to_rdd(paths, context = None, **kwargs):
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
    return path_rdd.map(lambda cpath: (cpath, ITKImage.read_itk_as_obj(cpath)._asdict()))

def read_to_dataframe(paths, context = None, **kwargs):
    """
    Create a dataframe from a list of paths, list of lists, or RDD of paths
    :param paths: files to load to make the ITK DataFrame
    :param context: SparkContext (only needed if paths is not an RDD)
    :param kwargs: additional arguments for creating the RDD with parallelize
    :return: DataFrame with all relevant fields see ITKImage
    """
    cur_rdd = _read_to_rdd(paths, context = context, **kwargs)
    return cur_rdd.map(lambda x: Row(filename = x[0], **x[1])).toDF()

def show_itk_image(img, title=None, margin=0.05, dpi=80 ):
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3,4):
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

        extent = (0, xsize*spacing[1], ysize*spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

        plt.set_cmap("gray")

        if z is None:
            ax.imshow(nda,extent=extent,interpolation=None)
        else:
            ax.imshow(nda[z,...],extent=extent,interpolation=None)

        if title:
            plt.title(title)

        plt.show()
        return fig

    try:
        from ipywidgets import interact, interactive
        from ipywidgets import widgets
    except ImportError:
        if slicer: print("Slicer can only be used inside of IPython")
        slicer = False

    if slicer:
        interact(lambda z: callback(z), z=(0,nda.shape[0]-1))
    else:
        return callback()

def show_itk_image3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))


    img_null = sitk.Image([0,0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        #TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            img = sitk.Compose(img_comps)
    return show_itk_image(img, title, margin, dpi)