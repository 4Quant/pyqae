"""
Modules related to Medicine and DICOM Files
"""
from dicom import read_file as dicom_simple_read
import numpy as np
from glob import glob
import pandas as pd

import dicom
from collections import namedtuple
from pyspark.sql.types import _infer_type, _has_nulltype, StructType, MapType, ArrayType
from pyspark.sql import Row, F
import pyspark.sql.types as sq_types

from PIL import Image as PImage
from matplotlib.pyplot import cm
import base64
from io import BytesIO


type_info = namedtuple('type_info', ['inferrable', 'realtype', 'has_nulltype', 'length', 'is_complex'])


def _tlen(x):
    """
    Try to calculate the length, otherwise return 1
    """
    try:
        return len(x)
    except:
        try:
            if np.isnan(x): return 0
        except:
            pass
        return 1


def _tnonempty(x):
    return _tlen(x) > 0


def safe_type_infer(x):
    COMPLEX_TYPES = (StructType, MapType)
    try:
        sq_type = _infer_type(x)
        return type_info(True, sq_type, has_nulltype=_has_nulltype(sq_type),
                         length=_tlen(x), is_complex=(type(sq_type) in COMPLEX_TYPES)
                         )
    except:
        return type_info(False, type(x), has_nulltype=False,
                         length=_tlen(x), is_complex=False)


def _identify_column_types(in_df_dict):
    return dict([(k, safe_type_infer(v)) for (k, v) in in_df_dict.items()])


def _findvalidvalues(crow):
    nz_vals = list(filter(_tnonempty, crow))
    return None if len(nz_vals) < 1 else nz_vals[0]


def _countmissingvalues(crow):
    nz_vals = list(filter(lambda i: not _tnonempty(i), crow))
    return len(nz_vals)


BAD_COLUMNS = tuple()


def dicom_to_dict(in_dicom, read_array=True):
    temp_dict = {a.name: a.value for a in in_dicom.iterall()}
    if in_dicom.__dict__.get('_pixel_array', None) is not None:
        temp_dict['Pixel Array'] = in_dicom.pixel_array.tolist()
    df_dicom = pd.DataFrame([temp_dict])  # just for the type conversion

    cur_kv_pairs = list(df_dicom.T.to_dict().values())[0]  # first row
    valid_keys = _identify_column_types(cur_kv_pairs)
    do_keep = lambda key, ti: ti.inferrable & (not ti.has_nulltype) & (not ti.is_complex)  # & (ti.length>0)
    fvalid_keys = dict([(k, do_keep(k, t_info)) for k, t_info in valid_keys.items()])
    return (dict([(k, v) for (k, v) in cur_kv_pairs.items() if fvalid_keys.get(k)]),
            dict([(k, v) for (k, v) in cur_kv_pairs.items() if not fvalid_keys.get(k)]),
            valid_keys)


def dicoms_to_dict(dicom_list):
    fvr = lambda x: None if x.first_valid_index() is None else x[x.first_valid_index()]

    out_list = []

    for in_dicom in dicom_list:
        temp_dict = {a.name: a.value for a in in_dicom.iterall()}
        if in_dicom.__dict__.get('_pixel_array', None) is not None:
            temp_dict['Pixel Array'] = in_dicom.pixel_array.tolist()

        out_list += [temp_dict]
    df_dicom = pd.DataFrame(out_list)  # just for the type conversion
    fvi_series = df_dicom.apply(_findvalidvalues, axis=0).to_dict()
    valid_keys = _identify_column_types(fvi_series)
    do_keep = lambda key, ti: ti.inferrable & (not ti.has_nulltype)  # & (not ti.is_complex) & (ti.length>0)
    fvalid_keys = dict([(k, do_keep(k, t_info)) for k, t_info in valid_keys.items()])
    good_columns = list(map(lambda x: x[0], filter(lambda x: x[1], fvalid_keys.items())))
    bad_columns = list(map(lambda x: x[0], filter(lambda x: not x[1], fvalid_keys.items())))
    sql_df = df_dicom[good_columns]
    return sql_df.dropna(axis=1)


def _remove_empty_columns(in_df):
    empty_cols = dict(filter(lambda kv: kv[1] > 0, in_df.apply(_countmissingvalues, axis=0).to_dict().items()))
    # remove missing columns
    return in_df[[ccol for ccol in in_df.columns if empty_cols.get(ccol, 0) == 0]]


# perform conversions
_dicom_conv_dict = {dicom.multival.MultiValue: lambda x: np.array(x).tolist(),
                    dicom.sequence.Sequence: lambda seq: [[(str(d_ele.tag), str(d_ele.value)) for d_ele in d_set] for
                                                          d_set in seq]}


def _apply_conv_dict(in_ele):
    cnv_fcn = _dicom_conv_dict.get(type(in_ele[0]), None)
    if cnv_fcn is not None:
        return in_ele.map(cnv_fcn)
    else:
        return in_ele


def _conv_df(in_df):
    return in_df.apply(_apply_conv_dict)


def dicom_paths_to_df(in_path_list):
    f_df = dicoms_to_dict([dicom_simple_read(in_path, stop_before_pixels=True) for in_path in in_path_list])
    f_df['DICOMPath4Q'] = in_path_list
    rec_df = _remove_empty_columns(f_df)
    conv_df = _conv_df(rec_df)
    return conv_df


_sq_conv_map = {
    np.int16: sq_types.IntegerType,
    np.uint8: sq_types.IntegerType,
    np.float32: sq_types.FloatType,
    np.float64: sq_types.DoubleType
}


def _ndarray_to_sql(in_arr):
    """

    _ndarray_to_sql(np.zeros((3,3,3), np.float32))
    >>> ArrayType(ArrayType(ArrayType(FloatType,true),true),true)

    _ndarray_to_sql(np.zeros((3,3,3), np.int16))
    >>> ArrayType(ArrayType(ArrayType(IntegerType,true),true),true)

    _ndarray_to_sql(np.zeros((3,3,3), np.object))
    Exception: object is not supported in SparkSQL
    """
    assert type(in_arr) is np.ndarray, "Only works for NDArrays"
    for (dtype, stype) in _sq_conv_map.items():
        if dtype == in_arr.dtype:
            base_type = stype()  # instantiate
            for c_dim in in_arr.shape:
                base_type = sq_types.ArrayType(base_type)
            return base_type

    raise Exception("{} is not supported in SparkSQL".format(in_arr.dtype))


twod_arr_type = sq_types.ArrayType(sq_types.ArrayType(sq_types.IntegerType()))
# the pull_input_tile function is wrapped into a udf to it can be applied to create the new image column
# numpy data is not directly supported and typed arrays must be used instead therefor we run the .tolist command
read_dicom_slice_udf = F.udf(lambda x: dicom_simple_read(x).pixel_array.tolist(), returnType=twod_arr_type)

def _np_to_uri(in_array, cmap='RdBu'):
    """

    _np_to_uri(np.zeros((100,100))
    >>> 'iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAABUElEQVR4nO3SQQEAEADAQBQRT/8ExPDYXYI9Ns/Yd5C1fgfwlwHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxBogzQJwB4gwQZ4A4A8QZIM4AcQaIM0CcAeIMEGeAOAPEGSDOAHEGiDNAnAHiDBBngDgDxBkgzgBxDxypAoX8C2RlAAAAAElFTkSuQmCC'
    """
    test_img_data = np.array(in_array).astype(np.float32)
    test_img_data -= test_img_data.mean()
    test_img_data /= test_img_data.std()
    test_img_color = cm.get_cmap(cmap)((test_img_data + 0.5).clip(0, 1))
    p_data = PImage.fromarray((test_img_color * 255).clip(0, 255).astype(np.uint8))
    rs_p_data = p_data.resize((128, 128), resample=PImage.BICUBIC)
    out_img_data = BytesIO()
    rs_p_data.save(out_img_data, format='png')
    out_img_data.seek(0) # rewind
    return base64.b64encode(out_img_data.read()).decode("ascii").replace("\n", "")

image_to_uri_udf = F.udf(_np_to_uri, returnType = sq_types.StringType())

_wrap_uri = lambda data_uri: "data:image/png;base64,{0}".format(data_uri)

def display_uri(uri_list):
    """

    show_uri(_np_to_uri(np.zeros((100,100))))

    """
    from IPython.display import HTML
    out_html = ""
    for in_uri in uri_list:
        out_html += """<img src="{0}" width = "100px" height = "100px" />""".format(_wrap_uri(in_uri))
    return HTML(out_html)

