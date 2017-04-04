import pytest
import os
pytestmark = pytest.mark.usefixtures("eng")
_res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'resources')
from pyqae.med import deepcrawler as dc

def test_deepcrawler(eng):
    all_tags=dc.read_tar_tags(os.path.join(_res_dir,'dicom.tar'))
    assert len(all_tags)==100, "length of tags should match"
