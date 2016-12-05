"""
doctests are significantly easier to run and maintain than the standard images

"""
if __name__ == "__main__":
    import doctest
    import pyqae
    doctest.testmod(pyqae, verbose = True)
    from pyqae import simplespark
    doctest.testmod(simplespark, verbose = True)
    from pyqae import sitk
    doctest.testmod(sitk, verbose=True)

    from pyqae.dnn import viz
    doctest.testmod(viz, verbose=True)

    from pyqae import simplespark
    doctest.testmod(simplespark)

    from pyqae import med
    doctest.testmod(med, verbose=True)

    from pyqae import viz
    doctest.testmod(viz, verbose=True)
