#!/usr/bin/env python

from setuptools import setup
from glob import glob
import os
package_files = ['pyqae']
package_files += [os.path.split(path)[0].replace("/", ".") for path in glob(
    os.path.join('pyqae', '*', '__init__.py'))]


version = '0.21'

setup(
    name='pyqae',
    version=version,
    description='large-scale image and time series analysis',
    author='fourquant',
    author_email='info@4quant.com',
    url='https://4quant.com',
    packages=package_files,
    install_requires=[line.replace('-gpu', '') for line in
                      open('requirements.txt').read().split('\n') if
                      'git+http' not in line],
    # skip all respositories since only pip knows how to install them
    # have tensorflow as a requirement not the GPU version (only for pip)
    long_description='See our main repository'
)
