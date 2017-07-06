__doc__ = """
A toolset for crawling through recursive directories of DICOM
files and archives
>>> n_ele=10000000
>>> n_ele=10
>>> gdict={}
>>> for i in range(n_ele): gdict['%012d' % i] = True
>>> for i in range(n_ele): _ = ('%012d' % i) in gdict
"""

MAX_VAL_LEN = 512  # don't need anything longer than 512 chars

try:
    from dicom import read_file as read_dicom_raw
except ImportError:
    # pydicom 1.0 changed names
    from pydicom import read_file as read_dicom_raw

import json
import os
from io import BytesIO
from tarfile import TarFile
from warnings import warn
import hashlib

def hash_path(path):
    """

    :param path:
    :return:
    >>> assert len(hash_path('/dir/md/bob'))<255
    >>> hash_path('/dir/md/bob')
    '75f2c6943b967a8e2568261f0abedf255ca824fe284453aabbbedb7c930edb413f3191a320020415c0fbf747edc54c42df36c66741227249701b7cddd90a1207'
    >>> hash_path('/dir/sd/bob')
    'c59559166a868b373dbcb47c8d89edbb3100392d8cecd4ce23b506cfc3d689f5f1d52b115e4354540994b904f4d00c1976994786621ed4d4d0f283507591054e'
    """
    str_path=str(path)
    path_asc=str_path.encode('ascii')
    return hashlib.sha512(path_asc).hexdigest()

import pandas as pd

_cur_dir = os.path.dirname(os.path.realpath(__file__))

_res_dir = os.path.join(os.path.realpath(os.path.join(_cur_dir, '..', '..',
                                                      'test')),
                        'resources')


def tlen(c_obj):
    try:
        return len(c_obj)
    except:
        return -1


def read_dicom_tags(fname, force=False):
    try:
        dcm_data = read_dicom_raw(fname, stop_before_pixels=True,
                                  force=force)
        return {a.name: a.value for a in
                dcm_data.iterall() if
                tlen(a.value) < MAX_VAL_LEN}
    except Exception as exc:
        warn(exc, RuntimeWarning)
        return None


def vread_dicom_tags(f_path, f_data):
    dcm_data = read_dicom_tags(BytesIO(f_data))
    dcm_list = list(dcm_data.items()) if dcm_data is not None else []
    return dict(dcm_list + [('path', f_path)])


def recursive_walk_path(base_path,
                        ext,
                        only_first=False, verbose=False):
    # type: (str, str, bool, bool) -> List[Tuple[str, str]]
    """
    Recursively browse a path and extract files which meet a criteria
    :param base_path:
    :param ext:
    :param only_first:
    :return:
    >>> recursive_walk_path(_res_dir,'tar')
    [('tar', '/Users/mader/Dropbox/Informatics/pyqae-master/test/resources/dicom.tar')]
    >>> df=recursive_walk_path(_res_dir,'dcm',True)
    >>> len(df)
    2
    >>> df[0][0]
    'dcm'
    >>> _=recursive_walk_path(_res_dir,'tar', verbose=True) #doctest: +NORMALIZE_WHITESPACE
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/dicom/subdir/subdir
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/dicom/subdir
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/dicom
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/multilayer_tif
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/nrrd
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/singlelayer_png
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources/singlelayer_tif
     Path: /Users/mader/Dropbox/Informatics/pyqae-master/test/resources
    """
    file_list = []
    for root, dirs, files in os.walk(base_path, topdown=False):
        if verbose: print('\t Path: {}'.format(root))
        c_files = [os.path.join(root, file) for file in
                   filter(lambda x: '.{}'.format(ext) in x and
                                    (not x.startswith('.')) and
                                    (not x.startswith('~')), files)]
        if only_first:
            if len(c_files) > 0:
                file_list += [c_files[0]]
        else:
            file_list += c_files
    return [(ext, c_file) for c_file in file_list]


def mrecursive_walk_path(base_path, ext_only_first_pairs):
    # type: (str, List[Tuple[str, bool]]) -> List[Tuple[str, str]]
    """
    Recursively browse a path and extract files which meet multiple criteria
    :param base_path:
    :param ext_only_first_pairs:
    :return:
    >>> mgen=mrecursive_walk_path(_res_dir,[('tar',False)])
    >>> [(ext,os.path.basename(path)) for (ext, path) in mgen]
    [('tar', 'dicom.tar')]
    >>> mgen=mrecursive_walk_path(_res_dir,[('dcm',True)])
    >>> [(ext,os.path.basename(path)) for (ext, path) in mgen]
    [('dcm', '1-051.dcm'), ('dcm', '10-060.dcm')]
    >>> mgen=mrecursive_walk_path(_res_dir,[('dcm',False)])
    >>> [(ext,os.path.basename(path)) for (ext, path) in mgen]
    [('dcm', '1-051.dcm'), ('dcm', '10-060.dcm'), ('dcm', '10-060.dcm')]
    """
    for root, dirs, files in os.walk(base_path, topdown=False):
        for ext, only_first in ext_only_first_pairs:
            n_files = filter(lambda x: '.{}'.format(ext) in x and
                                       (not x.startswith('.')) and
                                       (not x.startswith('~')), files)
            for c_file in n_files:
                yield (ext, os.path.join(root, c_file))
                if only_first:
                    break


def read_tar_tags(tar_path, show_matching_files=False):
    """
    Read the dicom from a tar file
    :param tar_path:
    :return: dictionary of key, value pairs
    >>> read_tar_tags(os.path.join(_res_dir,'dicom.tar'),True)
    ['dicom/10-060.dcm', 'dicom/subdir/subdir/1-051.dcm', 'dicom/subdir/subdir/10-060.dcm']
    >>> all_tags=read_tar_tags(os.path.join(_res_dir,'dicom.tar'))
    >>> len(all_tags)
    179
    >>> all_tags['path']
    '/Users/mader/Dropbox/Informatics/pyqae-master/test/resources/dicom.tar#dicom/10-060.dcm'
    >>> all_tags['Columns']
    512
    """
    with TarFile(tar_path, 'r') as c_tar:
        all_info = c_tar.getmembers()
        all_files = [tar_info for tar_info in all_info if
                     not tar_info.isdir() and
                     (tar_info.name.endswith('.dcm') and (
                         not os.path.basename(tar_info.name).startswith('.')))]
        if show_matching_files:
            return [cfile.name for cfile in all_files]
        if len(all_files) > 0:
            dcm_data = read_dicom_tags(c_tar.extractfile(all_files[0]))
            f_path = '{}#{}'.format(tar_path, all_files[0].name)
        else:
            warn("Must have at least one file", RuntimeWarning)
            dcm_data = None
            f_path = tar_path
        dcm_list = list(dcm_data.items()) if dcm_data is not None else []
        return dict(dcm_list + [('path', f_path)])


DEFAULT_READERS = {'tar': read_tar_tags, 'dcm': read_dicom_tags}
DEFAULT_WALK_ARGS = [('tar', False), ('dcm', True)]


def extract_files(in_paths,  # type: Union[List[str],str]
                  walk_args=None,  # type: Optional[List[Tuple[str,bool]]]
                  walk_readers=None,
                  # type: Optional[Dict[str,Callable[[str],Any]]]
                  max_results=None,
                  open_files=True
                  ):
    """

    :param in_paths:
    :param walk_args: the arguments for the recursive path
    :param walk_readers: the readers to use for parsing files
    :param max_results: maximum number of results to return
    :param open_files: open all of the files or just return the paths
    :return:
    >>> _test_paths=[_res_dir]
    >>> all_data=extract_files(_test_paths,open_files=False)
    >>> len(all_data)
    3
    >>> [(ext,os.path.basename(path)) for (ext, path) in all_data]
    [('dcm', '1-051.dcm'), ('dcm', '10-060.dcm'), ('tar', 'dicom.tar')]
    >>> all_data=extract_files(_test_paths,open_files=True)
    >>> len(all_data)
    3
    >>> len(all_data[0])
    178
    >>> all_data[0]['Image Type']
    ['ORIGINAL', 'PRIMARY', 'AXIAL']
    """
    out_gen = extract_files_gen(in_paths=in_paths,
                                walk_args=walk_args,
                                walk_readers=walk_readers,
                                open_files=open_files)
    if max_results is None:
        return list(out_gen)
    else:
        return [v for _, v in zip(range(max_results), out_gen)]


def extract_files_gen(in_paths,  # type: Union[List[str],str]
                      walk_args=None,  # type: Optional[List[Tuple[str,bool]]]
                      walk_readers=None,
                      # type: Optional[Dict[str,Callable[[str],Any]]]
                      open_files=True,
                      verbose=False
                      ):
    """

    :param in_paths:
    :param walk_args: the arguments for the recursive path
    :param walk_readers: the readers to use for parsing files
    :param use_generator: use a generator instead of a eager loop
    :param open_files: open all of the files or just return the paths
    :return:
    >>> all_data=extract_files_gen([_res_dir],open_files=False)
    >>> type(all_data)
    <class 'generator'>
    >>> [(ext,os.path.basename(path)) for (ext, path) in all_data]
    [('dcm', '1-051.dcm'), ('dcm', '10-060.dcm'), ('tar', 'dicom.tar')]
    >>> all_data=extract_files_gen(_res_dir,open_files=False)
    >>> type(all_data)
    <class 'generator'>
    >>> [(ext,os.path.basename(path)) for (ext, path) in all_data]
    [('dcm', '1-051.dcm'), ('dcm', '10-060.dcm'), ('tar', 'dicom.tar')]
    >>> all_data=extract_files_gen(_res_dir,open_files=True)
    >>> len(list(all_data))
    3
    """
    if type(in_paths) is str:
        in_paths = [in_paths]
    if walk_args is None:
        walk_args = DEFAULT_WALK_ARGS
    if walk_readers is None:
        walk_readers = DEFAULT_READERS
    for path in in_paths:
        n_files = mrecursive_walk_path(path,
                                       walk_args)
        for c_ext, c_file in n_files:
            if open_files:
                yield walk_readers[c_ext](c_file)
            else:
                yield (c_ext, c_file)


def write_json_pandas(path, c_dict):
    try:
        simple_df = pd.DataFrame([c_dict])
    except Exception as e:
        print('Failed pandas conversion {}'.format(e))
    try:
        simple_df.T[0].to_json(path, default_handler=lambda x: '')
    except Exception as e:
        print(e)
        for id, idval in c_dict.items():
            # print(id,'\t',type(idval),'\t',str(idval)[:10])
            test_val = simple_df[id].values[0]
            print(id, '\t', 'pd-type', type(test_val), 'raw-type', type(idval))
            try:
                print(len(json.dumps((id, test_val))))
            except:
                print('\t\tPDSerialization failed!!')
            try:
                print(len(json.dumps((id, idval))))
            except:
                print('\t\tRawSerialization failed!!')
            print(id, '\t', type(simple_df[id].values[0]))
            try:
                print(len(json.dumps((id, str(idval)))))
            except:
                print('\t\tStrSerialization failed!!')


def write_json_jsonlite(path, c_dict):
    with open(path, 'w') as f:
        json.dump([(k, str(v)) for k, v in c_dict.items()], f)


def extract_to_hjson(out_dir,
                     in_paths,
                     walk_args=None,  # type: Optional[List[Tuple[str,bool]]]
                     walk_readers=None, # type: Optional[Dict[str,Callable[[str],Any]]]
                     write=True):
    """
    Write to hashed paths instead of paths
    :param out_dir:
    :param in_paths:
    :param verbose:
    :param write:
    :return:
    >>> all_data=extract_to_hjson('fancy',_res_dir,write=False)
    >>> all_data[0]
    'fancy/7cdc0c37f7d98d7c5d33d12af920fec51ce4eac0681ff837460f55afcd2af83af1f8389eb09928b98fda144a207888792d69ec2c70e4f8c21a9b17369c0b5b65.json'
    >>> len(all_data)
    3
        """
    out_files=[]
    if not os.path.exists(out_dir):
        if write: os.mkdir(out_dir)
    for c_ext, c_file in mrecursive_walk_path(in_paths, DEFAULT_WALK_ARGS):
        path_name = os.path.join(out_dir,'%s.json' % (hash_path(c_file)))
        if not os.path.exists(path_name):
            c_dict=DEFAULT_READERS[c_ext](c_file)
            if write:
                write_json_jsonlite(path_name, c_dict)
            else:
                out_files += [path_name]
    return out_files


def extract_to_json(out_dir, in_paths, verbose=False, write=True):
    """

    :param out_dir:
    :param in_paths:
    :param verbose:
    :param write:
    :return:
    >>> all_data=extract_to_json('fancy',_res_dir,write=False)
    >>> all_data[0]
    '7731341270494673_000000000.json'
    >>> len(all_data)
    3
    """
    if not os.path.exists(out_dir):
        if write: os.mkdir(out_dir)
    out_files = []

    for i, c_dict in enumerate(extract_files_gen(in_paths,
                                                 open_files=True,
                                                 verbose=verbose)):

        path_name = '%s_%09d.json' % (c_dict.get('Accession Number', '-1'), i)
        if write:
            write_json_jsonlite(os.path.join(out_dir, path_name), c_dict)
        else:
            out_files += [path_name]
    return out_files


def _make_df(in_iter):
    ilist = list(in_iter)
    return [pd.DataFrame(ilist)] if len(ilist) > 0 else []


# the spark dependencies code

import warnings

try:
    # These are only available inside of the pyspark application (using py4j)
    from pyspark.sql import Row
    from pyspark.rdd import RDD
    from pyspark import SparkContext
    from pyspark.sql import SQLContext


    # spark tools
    def build_bf_pipe(sc, file_path, partitions=100):
        all_files_rdd = sc.binaryFiles(file_path, partitions)
        all_dicom_rdd = all_files_rdd.map(lambda x: vread_dicom_tags(*x))
        all_dicom_df = all_dicom_rdd.mapPartitions(
            lambda x: [pd.DataFrame(list(x))])
        return all_dicom_df


    def rdd_json_to_disk(df_rdd, out_dir):
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        df_rdd.zipWithUniqueId().foreach(
            lambda x: x[0].to_json(
                os.path.join(out_dir, '{}.json'.format(x[1]))))


    def pair_extract_files_spark(sc,
                            in_files,
                            walk_args=None,
                            # type: Optional[List[Tuple[str,bool]]]
                            walk_readers=None,
                            # type: Optional[Dict[str,Callable[[str],Any]]]
                            ):
        """

        :param sc:
        :param in_files:
        :param walk_args:
        :param walk_readers:
        :return:
        >>> raw_dicom, df_dicom = pair_extract_files_spark(sc,[_res_dir])
        >>> raw_dicom.count()
        880
        >>> df_dicom.count()
        80
        >>> all_tags=raw_dicom.first()
        >>> len(all_tags)
        100
        """
        if walk_args is None:
            walk_args = DEFAULT_WALK_ARGS
        if walk_readers is None:
            walk_readers = DEFAULT_READERS
        base_rdd = sc.parallelize(in_files)
        walk_args_rdd = sc.parallelize(walk_args)

        all_files_rdd = base_rdd.cartesian(walk_args_rdd).flatMap(
            lambda x: recursive_walk_path(x[0], x[1][0], x[1][1])).repartition(
            100)
        all_dicom_tags = all_files_rdd.map(lambda x: walk_readers[x[0]](x[1]))
        all_dicom_df = all_dicom_tags.mapPartitions(_make_df)
        return all_dicom_tags, all_dicom_df

    def gen_extract_files_spark(sc, in_files,
                                file_partitions=100,
                                dicom_partitions=10000
                                ):
        """

        :param sc:
        :param in_files:
        :param file_partitions:
        :param dicom_partions:
        :return:
        >>> c_rdd=gen_extract_files_spark([_res_dir])
        >>> type(c_rdd)
        >>> len(c_rdd)
        """
        base_rdd = sc.parallelize(in_files,
                                  file_partitions)
        all_tags_rdd = base_rdd.flatMap(
            lambda x: extract_files_gen(x)).repartition(dicom_partitions)
        return all_tags_rdd


except ImportError:
    warnings.warn("Pyspark is not available using simplespark backend instead",
                  ImportWarning)
    try:
        from pyqae.simplespark import Row
        from pyqae.simplespark import LocalRDD as RDD
        from pyqae.simplespark import LocalSparkContext as SparkContext
    except ImportError:
        warnings.warn('PYQAE Spark-replacement is also missing, no RDD-like '
                      'functionality available')

if __name__ == '__main__':
    import sys

    out_path, in_path = sys.argv[1:]
    print('Running deepcrawler on {}, {}'.format(out_path, in_path))
    extract_to_hjson(out_path, in_path)
