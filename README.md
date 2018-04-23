# pyqae [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/4Quant/pyqae/master)

> python/pyspark image query analysis engine

Pyqae is a python-based tool for processing
## install

The core `pyqae` package defines core data structures and read/write operations for image stacks

## Conda

Create a new environment using the environment file in the binder folder

```
conda env create -f binder/environment.yml
```

Install the remaining packages and tools using pip in the root directory of the package

```
pip install .
```

## Binder/Docker

You can use repo2docker to make a self-contained docker image directly from this repository
```
pip install jupyter-repo2docker
```

# Dry-run
You can see what will be built by performing a dry run in the local directory

```
repo2docker --debug --no-build .
```

or build and run the image using

```
repo2docker .
```



## other notes
It is built on [`numpy`](https://github.com/numpy/numpy), [`scipy`](https://github.com/scipy/scipy), [`scikit-learn`](https://github.com/scikit-learn/scikit-learn), and [`scikit-image`](https://github.com/scikit-image/scikit-image), and is compatible with Python 2.7+ and 3.4+. You can install it using:


The official procedure for installation is first running
```bash
pip install -r requirements.txt
```

And then running
```bash
python setup.py install
```

## related packages

There are a number of different tools which pyqae utilizes for analysis

- [`keras`](https://github.com/fchollet/keras) deep learning wrapper tools
- [`elephas`](https://github.com/maxpumperla/elephas) distribution code for ML and Keras
- [`tensorflow`](https://github.com/tensorflow/tensorflow) core deep learning code
- [`thunder`](https://github.com/thunder-project/thunder) thunder project for image and sequence analysis


You can install the ones you want with `pip`, for example

```
pip install thunder-python
```

## using with spark

Thunder doesn't require Spark and can run locally without it, but Spark and Thunder work great together! To install and configure a Spark cluster, consult the official [Spark documentation](http://spark.apache.org/docs/latest). Thunder supports Spark version 1.5+, and uses the Python API PySpark. If you have Spark installed, you can install Thunder just by calling `pip install thunder-python` on both the master node and all worker nodes of your cluster. Alternatively, you can clone this GitHub repository, and make sure it is on the `PYTHONPATH` of both the master and worker nodes.

Once you have a running cluster with a valid `SparkContext` — this is created automatically as the variable `sc` if you call the `pyspark` executable — you can pass it as the `engine` to any of Thunder's loading methods, and this will load your data in distributed `'spark'` mode. In this mode, all operations will be parallelized, and chained operations will be lazily executed.

## using notebooks with pyspark

```bash
PYSPARK_PYTHON=/Users/mader/anaconda/bin/python PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" /Applications/spark-2.1.1-bin-hadoop2.7/bin/pyspark --driver-memory 8g --master local[8]
```

### using an environment
```bash
PYSPARK_PYTHON=/Users/mader/anaconda/envs/py27/bin/python PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" /Applications/spark-2.1.1-bin-hadoop2.7/bin/pyspark --driver-memory 8g --master local[8]
```
# or the old version
```bash
PYSPARK_DRIVER_PYTHON=ipython PYSPARK_DRIVER_PYTHON_OPTS=notebook /Volumes/ExDisk/spark-2.0.0-bin-hadoop2.7/bin/pyspark
```

## contributing

Thunder is a community effort! The codebase so far is due to the excellent work of the following individuals:

> Andrew Osheroff, Ben Poole, Chris Stock, Davis Bennett, Jascha Swisher, Jason Wittenbach, Jeremy Freeman, Josh Rosen, Kunal Lillaney, Logan Grosenick, Matt Conlen, Michael Broxton, Noah Young, Ognen Duzlevski, Richard Hofer, Owen Kahn, Ted Fujimoto, Tom Sainsbury, Uri Laseron, W J Liddy

If you run into a problem, have a feature request, or want to contribute, submit an issue or a pull request, or come talk to us in the [chatroom](https://gitter.im/thunder-project/thunder)!
