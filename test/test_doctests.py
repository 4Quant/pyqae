"""
doctests are significantly easier to run and maintain than the standard images

"""
import doctest
import pytest
def test_pyqae():
    import pyqae
    doctest.testmod(pyqae, verbose=True)

def test_simplespark():
    from pyqae import simplespark
    doctest.testmod(simplespark, verbose=True)

def test_sitk():
    from pyqae import sitk
    doctest.testmod(sitk, verbose=True)

def test_viz():
    from pyqae.dnn import viz
    doctest.testmod(viz, verbose=True)

def test_med():
    from pyqae import med
    doctest.testmod(med, verbose=True)

def test_seg():
    from pyqae.images import segmentation
    doctest.testmod(segmentation, verbose=True)

def test_nd():
    from pyqae import nd
    doctest.testmod(nd, verbose=True)

def test_skl():
    from pyqae import skl
    doctest.testmod(skl, verbose=True)

def test_nlp():
    from pyqae import nlp
    doctest.testmod(nlp, verbose=True)


if __name__ == "__main__":
    test_nd()
    test_seg()
    test_pyqae()
    test_simplespark()
    test_sitk()
    test_viz()
    test_med()
    test_nlp()



