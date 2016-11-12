import logging
import os
from glob import glob
import json
import numpy as np

class TypeTool(object):
    """
    For printing type outputs with a nice format
    """
    @staticmethod
    def info(obj):
        """
        Produces pretty text descriptions of type information from an object
        :param obj:
        :return: str for type
        """
        if type(obj) is tuple:
            return '({})'.format(', '.join(map(TypeTool.info,obj)))
        elif type(obj) is list:
            return 'List[{}]'.format(TypeTool.info(obj[0]))
        else:
            ctype_name = type(obj).__name__
            if ctype_name == 'ndarray': return '{}[{}]{}'.format(ctype_name,obj.dtype, obj.shape)
            elif ctype_name == 'str': return 'string'
            elif ctype_name == 'bytes': return 'List[byte]'
            else: return ctype_name


def local_read_depth(in_folder, depth, ext = '.dcm', inc_parent = False):
    """
    Read recursively from a list of directories
    :param in_folder: the base path to start from
    :param depth: the depth to look in the tree
    :param ext: the extension to search for
    :param inc_parent: to include the results from parent directories as well
    :return: a list of files
    """
    c_path = [in_folder]
    out_files = []
    for i in range(depth+1):
        c_wc_path = os.path.join(*(c_path + ['*']*i + ['*'+ext]))
        out_files += [] if (not inc_parent) and (i<depth) else glob(c_wc_path)
    return out_files

def _fix_col_names(t_prev_df, rep_char = ""):
    """
    Fix the column names to remove invalid characters
    :param t_prev_df: the old table
    :param rep_char: the character to replace bad characters with
    :return: a table with identical content but with the bad characters deleted from column names
    """
    new_df = t_prev_df
    for col in t_prev_df.columns:
        new_col = col
        for fix_chr in ' ,;{}()\n\t=':
            new_col = rep_char.join(new_col.split(fix_chr))
        new_df = new_df.withColumnRenamed(col, new_col)
    return new_df


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    A JSON plugin that allows numpy data to be serialized 
    correctly (if inefficiently)
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray): # and obj.ndim == 1:
            return obj.tolist()
        if isinstance(obj, np.number):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def notsupported(mode):
    logging.getLogger('pyqae').warn("Operation not supported in '%s' mode" % mode)
    pass

def check_spark():
    SparkContext = False
    try:
        from pyspark import SparkContext
    finally:
        return SparkContext

def check_options(option, valid):
    if option not in valid:
        raise ValueError("Option must be one of %s, got '%s'" % (str(valid)[1:-1], option))

def check_path(path, credentials=None):
    """
    Check that specified output path does not already exist

    The ValueError message will suggest calling with overwrite=True;
    this function is expected to be called from the various output methods
    that accept an 'overwrite' keyword argument.
    """
    from pyqae.readers import get_file_reader
    reader = get_file_reader(path)(credentials=credentials)
    existing = reader.list(path, directories=True)
    if existing:
        raise ValueError('Path %s appears to already exist. Specify a new directory, '
                         'or call with overwrite=True to overwrite.' % path)

def connection_with_anon(credentials, anon=True):
    """
    Connect to S3 with automatic handling for anonymous access.

    Parameters
    ----------
    credentials : dict
        AWS access key ('access') and secret access key ('secret')

    anon : boolean, optional, default = True
        Whether to make an anonymous connection if credentials fail to authenticate
    """
    from boto.s3.connection import S3Connection
    from boto.exception import NoAuthHandlerFound

    try:
        conn = S3Connection(aws_access_key_id=credentials['access'],
                            aws_secret_access_key=credentials['secret'])
        return conn

    except NoAuthHandlerFound:
        if anon:
            conn = S3Connection(anon=True)
            return conn
        else:
            raise

def connection_with_gs(name):
    """
    Connect to GS
    """
    import boto
    conn = boto.storage_uri(name, 'gs')
    return conn