#!/usr/bin/env python

from setuptools import setup
from glob import glob
import os
package_files = ['pyqae'] \
                + [os.path.split(path)[0].replace("/", ".") for path in glob(os.path.join('pyqae', '*', '__init__.py'))]


version = '0.20'

setup(
    name='pyqae-python',
    version=version,
    description='large-scale image and time series analysis',
    author='fourquant',
    author_email='info@4quant.com',
    url='https://4quant.com',
    packages=package_files,
    install_requires=open('requirements.txt').read().split('\n'),
    long_description='See our main repository'
)
