import numpy as np
from dicom import read_file as dicom_simple_read
from glob import glob
from collections import namedtuple
import os

DicomSlice = namedtuple('DicomSlice',['PixelSpacing','SliceLocation','SliceThickness','pixel_array'])

def read_dicom_slice(file_name):
    """
    Read a Dicom as a DicomSlice instead of the less flexible standard format
    """
    t_img = dicom_simple_read(file_name)
    return DicomSlice(np.array(t_img.PixelSpacing), float(t_img.SliceLocation), float(t_img.SliceThickness), t_img.pixel_array)

def read_dicom_list(file_list):
    """
    Read a list of dicoms and return a sorted stack and a tensor
    """
    all_slices = [read_dicom_slice(cfile) for cfile in file_list]
    all_slices = sorted(all_slices, key = lambda x: -1*x.SliceLocation)
    return all_slices

def read_dicom_tensor(file_list):
    """
    Read a list of dicom files into a tensor and wrap the results in a DicomSlice object
    """
    all_slices = read_dicom_list(file_list)
    return DicomSlice(all_slices[0].PixelSpacing, all_slices[0].SliceLocation, all_slices[0].SliceThickness,
                      np.stack([np.expand_dims(cslice.pixel_array,0) for cslice in all_slices]))
