#!/usr/bin/env python

from setuptools import setup

version = '0.1'

setup(
    name='pyqae-python',
    version=version,
    description='large-scale image and time series analysis',
    author='fourquant',
    author_email='info@4quant.com',
    url='https://4quant.com',
    packages=[
        'pyqae',
        'pyqae.blocks',
        'pyqae.dnn',
        'pyqae.nd',
        'pyqae.med',
        'pyqae.images',
        'pyqae.rddviz'
    ],
    install_requires=open('requirements.txt').read().split('\n'),
    long_description='See our main repository'
)
