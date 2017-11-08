"""Tools for dealing with HDF5 data efficiently"""
import collections
import os
from glob import glob
from warnings import warn

import h5py
import numpy as np
from tqdm import tqdm


def write_df_as_hdf(out_path, out_df, compression='gzip'):
    with h5py.File(out_path, 'w') as h:
        for k, arr_dict in tqdm(out_df.to_dict().items()):
            try:
                s_data = np.stack(arr_dict.values(), 0)
                try:
                    h.create_dataset(k, data=s_data, compression=
                    compression)
                except TypeError as e:
                    try:
                        h.create_dataset(k, data=s_data.astype(np.string_),
                                         compression=compression)
                    except TypeError as e2:
                        print('%s could not be added to hdf5, %s' % (
                            k, repr(e), repr(e2)))
            except ValueError as e:
                print('%s could not be created, %s' % (k, repr(e)))
                all_shape = [np.shape(x) for x in arr_dict.values()]
                warn('Input shapes: {}'.format(all_shape))


def create_or_expand_dataset(h5_file,  # type: h5py.File
                             ds_name,  # type: str
                             sample_count,  # type: int
                             sample_shape,  # type: Tuple[int, int, int]
                             dtype,  # type: Union[np.type, str]
                             compression='gzip'):
    # type: (...) -> (h5py.Dataset, int)
    """
    The function creates or expands a dataset within an hdf5 file. Particularly useful for making HDF5Matrix files
    for use with Keras. It allows iterative expansion of a dataset while preserving compression and chunking performance
    """
    if ds_name not in h5_file:
        start_idx = 0
        out_ds = h5_file.create_dataset(ds_name,
                                        shape=(sample_count,) + sample_shape,
                                        dtype=dtype,
                                        chunks=(sample_count,) + sample_shape,
                                        maxshape=(None,) + sample_shape,
                                        compression=compression)
    else:
        out_ds = h5_file[ds_name]
        assert out_ds.shape[
               1:] == sample_shape, 'All remaining dimensions should match {}!={}'.format(
            out_ds.shape,
            sample_shape)

        start_idx = out_ds.shape[0]
        out_ds.resize(sample_count + start_idx, axis=0)
    return out_ds, start_idx


class HDF5MatrixFolder:
    """Representation of HDF5 dataset to be used instead of a Numpy array.

    This is a replacement for the `HDF5Matrix` class in `keras`. It can
    represent a numpy array spread over several HDF5 files instead of just one.

    # Example
    ```python
        x_data = HDF5Matrix('input/', 'data')
        model.predict(x_data)
    ```
    # More Examples
    >>> from tempfile import TemporaryDirectory
    >>> from pyqae.utils import get_error, pprint
    >>> o_dir = TemporaryDirectory()
    >>> o_path = o_dir.name
    >>> get_error(HDF5MatrixFolder, datapath = o_path, dataset = 'data')
    'No matching hdf5 files found or provided, suffix : h5'
    >>> c_path = lambda x: os.path.join(o_path, 'out_%04d.h5' % x)
    >>> t_file = h5py.File(c_path(0), 'w')
    >>> t_file.create_dataset('data', data = np.zeros((10, 5, 4)))
    <HDF5 dataset "data": shape (10, 5, 4), type "<f8">
    >>> t_file.close()
    >>> t_file = h5py.File(c_path(1), 'w')
    >>> t_file.create_dataset('data', data = np.ones((10, 5, 4)))
    <HDF5 dataset "data": shape (10, 5, 4), type "<f8">
    >>> t_file.close()
    >>> hmat = HDF5MatrixFolder(o_path, 'data')
    >>> hmat.shape
    (20, 5, 4)
    >>> hmat[:].shape # check reading everything works
    (20, 5, 4)
    >>> np.shape(hmat[0:20]) # check standard reading works
    (20, 5, 4)
    >>> np.shape(hmat[np.arange(20)]) # check that numpy arrays work
    (20, 5, 4)
    >>> len(hmat)
    20
    >>> pprint(hmat[0])
    [[[ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]]]
    >>> pprint(hmat[19])
    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]
    >>> from keras import backend as K
    >>> K.set_image_dim_ordering('tf')
    >>> from keras.models import Sequential
    >>> from keras.layers import Activation
    >>> test_model = Sequential()
    >>> test_model.add(Activation('linear', input_shape = (5,4)))
    >>> test_model.predict(hmat, verbose = False).shape
    (20, 5, 4)
    >>> test_model.compile(optimizer = 'adam', loss = 'mse')
    >>> _ = test_model.fit(hmat, hmat, epochs=1, verbose = False)
    >>> o_dir.cleanup()
    """

    def __init__(self,
                 datapath,
                 dataset,
                 start=0,
                 end=None,
                 normalizer=None,
                 suffix="h5"):
        self.shapes = {}
        # used to avoid opening a file more than once
        self.refs = {}

        if isinstance(datapath, str):
            self.fnames = glob(os.path.join(datapath, "*{}".format(suffix)))
        elif isinstance(datapath, collections.Iterable):
            self.fnames = datapath
        else:
            raise RuntimeError("datapath argument has to be a list of paths "
                               "or glob pattern, got %s." % datapath)

        self.fnames = list(sorted(self.fnames))
        if len(self.fnames) < 1:
            raise ValueError('No matching hdf5 files found or provided, '
                             'suffix : {}'.format(suffix))

        self.dataset = dataset

        for fname in self.fnames:
            if fname not in self.refs:
                f = h5py.File(fname)
                self.refs[fname] = f

            f = self.refs[fname]

            if dataset not in f:
                raise RuntimeError("File (%s) does not contain a dataset "
                                   "named %s" % (fname, dataset))

            ds = f[dataset]
            self.shapes[fname] = ds.shape

        self.start = start
        if end is None:
            self.end = sum(shape[0] for shape in self.shapes.values())
        else:
            self.end = end

        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.end
            if stop + self.start <= self.end:
                idx = slice(start + self.start, stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key + self.start < self.end:
                idx = slice(key + self.start, key + self.start + 1)
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
                idx = slice(np.min(idx), np.max(idx) + 1)
            else:
                raise IndexError
        elif isinstance(key, (list, tuple)):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
                idx = slice(np.min(idx), np.max(idx) + 1)
            else:
                raise IndexError
        else:
            raise IndexError

        combined = []
        i = 0
        start, stop = idx.start, idx.stop
        for fname in self.fnames:
            data = self.refs[fname][self.dataset]
            combined.append(data[start:stop])
            i += data.shape[0]
            if start is not None:
                start -= data.shape[0]
                if start <= 0:
                    start = None
            stop -= data.shape[0]
            if stop <= 0:
                break

        data = np.vstack(combined)
        if self.normalizer is not None:
            return self.normalizer(data)
        else:
            return data

    @property
    def shape(self):
        for c_shape in self.shapes.values():
            return (self.end - self.start,) + c_shape[1:]
