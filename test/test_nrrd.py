import pytest
import os
import glob
import json
import nrrd
pytestmark = pytest.mark.usefixtures("eng")

resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
nrrd_path = os.path.join(resources, "nrrd")

def test_loadseg():
    seg_file = os.path.join(nrrd_path, "Tumor.seg.nrrd")
    data, options = nrrd.read(seg_file)
    print('seg keys', list(options.keys()))
    assert data.shape == (13, 10, 22), "Data shape is incorrect {}".format(data.shape)
    assert data.min() == 0, "Min Value is incorrect"
    assert data.max() == 1, "Max Value is incorrect"
    assert options['space origin'] == ['46.484375', '-13.671875', '-250.37695312499997']
    assert options['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset'] == '55 66 156'

def test_loadctdata():
    seg_file = os.path.join(nrrd_path, "402 WB_3D.nrrd")
    data, options = nrrd.read(seg_file)
    print('seg keys', list(options.keys()))
    assert data.shape == (128, 128, 277), "Data shape is incorrect {}".format(data.shape)
    assert round(data.min()) == 0, "Min Value is incorrect {}".format(data.min())
    assert round(data.max()) == 70358, "Max Value is incorrect {}".format(data.max())
    assert options['space origin'] == ['-347.26562500000006', '-347.26562500000006', '-760.49999999999989'], "Space origin does not match {}".format(options['space origin'])

if __name__ == "__main__":
    test_loadseg()
    test_loadctdata()
