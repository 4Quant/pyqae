import json
import logging
import os
from glob import glob
from tempfile import NamedTemporaryFile

import numpy as np

try:
    from typing import Tuple, List, Optional, Union, Dict, Any
except ImportError:
    print("List from typing is missing but not really needed")


    # junk variables
    # noinspection PyPep8Naming
    def List():
        raise RuntimeError("This should not be used for anything")


    Tuple = List
    Optional = List
    Any = List
    Union = List
    Dict = List

try:
    from tqdm import tqdm as fancy_progress_bar
except ImportError:
    # tqdm not installed
    fancy_progress_bar = lambda x: x


def get_temp_filename(suffix):
    with NamedTemporaryFile(suffix=suffix) as w:
        return w.name


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
            return '({})'.format(', '.join(map(TypeTool.info, obj)))
        elif type(obj) is list:
            return 'List[{}]'.format(TypeTool.info(obj[0]))
        else:
            ctype_name = type(obj).__name__
            if ctype_name == 'ndarray':
                return '{}[{}]{}'.format(ctype_name, obj.dtype, obj.shape)
            elif ctype_name == 'str':
                return 'string'
            elif ctype_name == 'bytes':
                return 'List[byte]'
            else:
                return ctype_name


def local_read_depth(in_folder, depth, ext='.dcm', inc_parent=False):
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
    for i in range(depth + 1):
        c_wc_path = os.path.join(*(c_path + ['*'] * i + ['*' + ext]))
        out_files += [] if (not inc_parent) and (i < depth) else glob(
            c_wc_path)
    return out_files


def _fix_col_names(t_prev_df, rep_char=""):
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


try:
    # python2
    from Queue import Queue, Empty as queueEmpty
except ImportError:
    # python3
    from queue import Queue, Empty as queueEmpty

from threading import Thread
from itertools import chain

pprint = lambda x, p=2: print(np.array_str(x, max_line_width=80, precision=
p))

def get_error(f, **kwargs):
    try:
        f(**kwargs)
        return "No Error"
    except Exception as e:
        return "{}".format(e)


def pqueue_map(in_func, in_plist, threads=None):
    in_list = [x for x in in_plist]
    q_in = Queue(len(in_list))
    q_out = Queue(len(in_list))
    threads = len(in_list) if threads is None else threads

    def _ex_fun():
        while True:
            try:
                x = q_in.get(block=False)
                q_out.put(in_func(x))
                q_in.task_done()
            except queueEmpty:
                break

    for x in in_list:
        q_in.put(x)
    # start threads
    r_threads = [Thread(target=_ex_fun) for i in range(threads)]
    _ = list(map(lambda t: t.start(), r_threads))

    q_in.join()
    _ = [t.join() for t in r_threads]
    out_list = [q_out.get_nowait() for _ in range(0, q_out.qsize())]
    return out_list


def pqueue_flatmap(in_func, in_list, threads=None):
    out_res = chain(*pqueue_map(in_func, in_list))
    return list(out_res)


def pqueue_flatmapvalues(in_func, in_plist, threads=None):
    in_list = [x for x in in_plist]
    q_in = Queue(len(in_list))
    q_out = Queue(len(in_list))
    threads = len(in_list) if threads is None else threads

    def _ex_fun():
        while True:
            try:
                in_key, in_val = q_in.get(block=False)
                q_out.put([(in_key, pval) for pval in in_func(in_val)])
                q_in.task_done()
            except queueEmpty:
                break

    # populate queue
    for x in in_list:
        q_in.put(x)
    # start threads
    r_threads = [Thread(target=_ex_fun) for i in range(threads)]
    _ = list(map(lambda t: t.start(), r_threads))
    # join threads
    _ = [t.join() for t in r_threads]

    # make outputs
    out_list = []
    for _ in range(0, q_out.qsize()):
        out_list += q_out.get_nowait()
    return out_list


def show_partition_sizes(in_rdd):
    return in_rdd.mapPartitionsWithIndex(
        lambda i, x_list: [[(i, len(list(x_list)))]]).collect()


def threaded_map(in_rdd, in_operation, threads_per_worker=None):
    """
    Run an io-bound map operation on multiple threads using queue and thread in python

    :param in_rdd:
    :param in_operation: the command to run
    :param threads_per_worker: number of threads to run on each worker (default is unlimited)
    :return:
    """

    if threads_per_worker == 1: return in_rdd.map(in_operation)

    def _part_flatmap(x_list):
        return pqueue_map(in_operation, x_list, threads=threads_per_worker)

    return in_rdd.mapPartitions(_part_flatmap)


def threaded_flatmap(in_rdd, in_operation, threads_per_worker=None):
    """
    Run an io-bound flatmap operation on multiple threads using queue and thread in python

    :param in_rdd:
    :param in_operation: the command to run
    :param threads_per_worker: number of threads to run on each worker (default is unlimited)
    :return:
    """

    if threads_per_worker == 1: return in_rdd.flatMap(in_operation)

    def _part_flatmap(x_list):
        return pqueue_flatmap(in_operation, x_list, threads=threads_per_worker)

    return in_rdd.mapPartitions(_part_flatmap)


def threaded_flatmapvalues(in_rdd, in_operation, threads_per_worker=None):
    """
    Run an io-bound flatmap operation on multiple threads using queue and thread in python

    :param in_rdd:
    :param in_operation: the command to run
    :param threads_per_worker: number of threads to run on each worker (default is unlimited)
    :return:
    """

    if threads_per_worker == 1: return in_rdd.flatMapValues(in_operation)

    def _part_flatmapvalues(x_list):
        return pqueue_flatmapvalues(in_operation, x_list,
                                    threads=threads_per_worker)

    return in_rdd.mapPartitions(_part_flatmapvalues)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    A JSON plugin that allows numpy data to be serialized 
    correctly (if inefficiently)
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        if isinstance(obj, np.number):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# dictionary tools
filter_dict = lambda fcn, old_dict: dict(filter(fcn, old_dict.items()))
dict_append = lambda idct, new_kvs: dict(list(idct.items()) + new_kvs)
dict_kwappend = lambda idct, **new_kvs: dict(
    list(idct.items()) + list(new_kvs.items()))

from warnings import warn


def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """

    attributes_list = []

    for i, test_attribute in enumerate(dir(im_props[0])):

        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        try:
            if test_attribute[:1] != '_' and not \
                    isinstance(getattr(im_props[0], test_attribute),
                               np.ndarray):
                attributes_list += [test_attribute]
        except Exception as e:
            warn("Not implemented: {} - {}".format(test_attribute, e),
                 RuntimeWarning)

    return attributes_list


def notsupported(mode):
    logging.getLogger('pyqae').warn(
        "Operation not supported in '%s' mode" % mode)
    pass


def check_spark():
    SparkContext = False
    try:
        from pyspark import SparkContext
    finally:
        return SparkContext


def check_options(option, valid):
    if option not in valid:
        raise ValueError(
            "Option must be one of %s, got '%s'" % (str(valid)[1:-1], option))


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
        raise ValueError(
            'Path %s appears to already exist. Specify a new directory, '
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
