import numpy as np
from skimage.io import imread
import pandas as pd
from io import BytesIO, StringIO
try:
    from dicom import read_dicom_file
except:
    def read_dicom_file(*args, **kwargs):
        raise Exception("Dicom Library is not available")

def _setup():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(name)s] %(levelname)s %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

_setup()

from glob import glob

__version__ = '0.1'

class PyqaeContext(object):
    """
    The primary context for performing PYQAE functions
    """
    def __init__(self, cur_sc = None, faulty_io = 'FAIL', retry_att = 5, *args, **kwargs):
        """
        Create or initialize a new Pyqae Context
        
        Parameters
        ----------
        cur_sc : SparkContext
            An existing initialized SparkContext, if none a new one is initialized with the other parameters.
        faulty_io : String
            A string indicating what should happen if a file is missing (FAIL, RETRY, or return an EMPTY value)
        retry_att : Int
            The number of times a retry should be attempted (if faulty_io is in mode RETRY otherwise ignored)
        """
        assert faulty_io in ['FAIL', 'RETRY', 'EMPTY'], "Faulty IO must be in the list of FAIL, RETRY, or EMPTY"
        assert retry_att>0, "Retry attempt must be greater than 0"
        self.faulty_io = faulty_io
        self.retry_att = retry_att
        if cur_sc is None: 
            from pyspark import SparkContext
            self._cur_sc = SparkContext(*args, **kwargs)
        else:
            self._cur_sc = cur_sc
    
    @staticmethod
    def _wrapIOCalls(method, faulty_io, retry_att):
        """
        A general wrapper for IO calls which should be retried or returned empty 
        
        """
        assert faulty_io in ['FAIL', 'RETRY', 'EMPTY']
        assert retry_att > 0, "Retry attempts should be more than 0, {}".format(retry_att)
        if faulty_io == 'FAIL':
            return method
        else:
            def wrap_method(*args, **kwargs):
                if faulty_io == 'RETRY': max_iter = retry_att-1
                else: max_iter = 1
                
                for i in range(max_iter):
                    try:
                        return method(*args,**kwargs)
                    except:
                        if faulty_io == 'EMPTY': return None
                # if it still hasn't passed throw the error
                return method(*args,**kwargs)
            return wrap_method
    
    @staticmethod
    def readBinaryBlobAsImageArray(iblob):
        return imread(BytesIO(iblob))
    
    @staticmethod
    def readBinaryBlobAsDicomArray(iblob):
        sio_blob = BytesIO(iblob)
        return read_dicom_file(sio_blob)
    
    @staticmethod
    def imageTableToDataFrame(imt_rdd):
        return imt_rdd.map(lambda x: dict(list(x[0].iteritems())+[('image_data',x[1].tolist())])).toDF()

    def readImageDirectoryLocal(self, path, parts=100, imread_fcn = imread):
        """
        Read a directory of images where the path is local / shared file system

        Parameters
        ----------
        path : String
            A path with wildcards for the images files, must be a shared directory accessible to all nodes
        path : List[String]
            A path can also be a list of strings
        """
        read_fun = PyqaeContext._wrapIOCalls(imread_fcn, self.faulty_io, self.retry_att)
        image_list = glob(path) if not isinstance(path, list) else path
        assert len(image_list)>0, "Image List cannot be empty"
        return self._cur_sc.parallelize(image_list, parts).map(lambda x: (x, read_fun(x)))

    def readImageDirectory(self, path, parts = 100):
        """
        Read a directory of images
        
        Parameters
        ----------
        path : String
            A path with wildcards for the images files can be prefixed with (s3, s3a, or a shared directory)
        """
        read_fun = PyqaeContext._wrapIOCalls(PyqaeContext.readBinaryBlobAsImageArray, self.faulty_io, self.retry_att)
        return self._cur_sc.binaryFiles(path, parts).mapValues(read_fun)
    
    def readDicomDirectory(self, path, parts = 100):
        """
        Read a directory of dicom files
        
        Parameters
        ----------
        path : String
            A path with wildcards for the images files can be prefixed with (s3, s3a, or a shared directory)
        """
        read_fun = PyqaeContext._wrapIOCalls(PyqaeContext.readBinaryBlobAsDicomArray, self.faulty_io, self.retry_att)
        return self._cur_sc.binaryFiles(path, parts).mapValues(read_fun)
    
    def readImageTable(self, path, col_name, im_path_prefix = '', parts = 100, 
                       read_table_func = pd.read_csv, preproc_func = None):
        """
        Read a table from images from a csv file
        
        Parameters
        ----------
        path : String
            A path to the csv file
        col_name : String
            The name of the column containing the path to individual images
        im_path_prefix : String
            The prefix to append to the path in the text file so it is opened correctly (default empty)
        read_table_func: Function (String -> Pandas DataFrame)
            The function to read the table from a file-buffer object (default is the read_csv function)
        preproc_func: Function (ndarray -> ndarray)
            A function to preprocess the image (filtering, resizing, padding, etc)
        """
        c_file = self._cur_sc.wholeTextFiles(path,1)
        assert c_file.count()==1, "This function only support a single file at the moment"
        full_table_buffer = StringIO("\n".join(c_file.map(lambda x: x[1]).collect()))
        image_table = read_table_func(full_table_buffer)
        image_paths = [os.path.join(im_path_prefix,cpath) for cpath in image_table[col_name]]
        # read the binary files from a list
        rawimg_rdd = self._cur_sc.binaryFiles(",".join(image_paths),parts)
        read_fun = PyqaeContext._wrapIOCalls(PyqaeContext.readBinaryBlobAsImageArray, self.faulty_io, self.retry_att)
        img_rdd = rawimg_rdd.mapValues(read_fun)
        pp_img_rdd = img_rdd if preproc_func is None else img_rdd.mapValues(preproc_func)
        # add the file prefix so the keys come up in the map operation
        image_paths = ['file:{}'.format(cpath) if cpath.find(':')<0 else cpath for cpath in image_paths]
        image_list = dict(zip(image_paths,image_table.T.to_dict().values()))
        
        return img_rdd.map(lambda x: (image_list[x[0]],x[1]))
    
    @staticmethod
    def _imageTableToDataFrame(imd_rdd, cur_sql, img_key_name = "image"):
        """
        Converts an image table to a DataFrame by converting the ndarray into a nested list (inefficient)
        but necessary for JVM compatibility. Written as a staticmethod to encapsulate the serialization.
        #TODO implement ndarray <-> JVM exchange
        
        Parameters
        ----------
        imd_rdd: RDD[(dict[String,_], ndarray)]
            The imageTable (created by readImageTable)
        
        cur_sql: SQLContext
            The SQLContext in which to make the DataFrame (important for making tables later)
        
        """
        first_row_dict, _ = imd_rdd.first()
        im_tbl_keys = list(first_row_dict.keys())
        #TODO handle key missing errors more gracefully
        iml_rdd = imd_rdd.map(lambda kv_pair: [kv_pair[0].get(ikey) for ikey in im_tbl_keys]+[kv_pair[1].tolist()])
        return cur_sql.createDataFrame(iml_rdd, im_tbl_keys+[img_key_name])
    
    def readImageDataFrame(self, path, col_name, im_path_prefix = '', parts = 100, 
                           read_table_func = pd.read_csv, preproc_func = None,
                          sqlContext = None):
        """
        Read a table from images from a csv file and return as a dataframe
        See Help from [[readImageTable]]
        Parameters
        ----------
        
        sqlContext: SQLContext
            The SQL context to use (if one exists) otherwise make a new one
        """
        imd_rdd = self.readImageTable(path, col_name, im_path_prefix = im_path_prefix, 
                                   parts = parts, read_table_func = read_table_func)
        cur_sql = sqlContext if sqlContext is not None else SQLContext(self._cur_sc)
        return PyqaeContext._imageTableToDataFrame(imd_rdd, cur_sql)