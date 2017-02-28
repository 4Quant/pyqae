FROM debian:jessie

MAINTAINER bbp_user <bluebrainproject@4quant.com>

ENV DEBIAN_FRONTEND noninteractive
ENV CONDA_DIR /opt/conda

# Core installs
RUN apt-get update && \
    apt-get install -y git vim wget build-essential python-dev ca-certificates bzip2 libsm6 && \
    apt-get clean

# Install conda
RUN echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda install --yes conda==4.2.9

# Create a user
RUN useradd -m -s /bin/bash bbp_user
RUN chown -R bbp_user:bbp_user $CONDA_DIR

# Open port
EXPOSE 8888

# Env vars
USER bbp_user
ENV HOME /home/bbp_user
ENV SHELL /bin/bash
ENV USER bbp_user
ENV PATH $CONDA_DIR/bin:$PATH
WORKDIR $HOME

# Setup ipython
RUN conda install --yes ipython-notebook terminado && conda clean -yt
RUN ipython profile create

# Switch to root for permissions
USER root

# Java setup
RUN apt-get install -y default-jre

# Spark setup
## Spark dependencies
ENV APACHE_SPARK_VERSION 2.0.1
ENV PYJ_VERSION py4j-0.10.1-src.zip
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends openjdk-7-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
        wget -q http://d3kbcqa49mib13.cloudfront.net/spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz && \
        tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz -C /usr/local && \
        rm spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz
RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6 spark

# Spark and Mesos config
ENV SPARK_HOME /usr/local/spark
ENV PATH $PATH:$SPARK_HOME/bin
RUN sed 's/log4j.rootCategory=INFO/log4j.rootCategory=ERROR/g' $SPARK_HOME/conf/log4j.properties.template > $SPARK_HOME/conf/log4j.properties
ENV _JAVA_OPTIONS "-Xms512m -Xmx4g"
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/$PYJ_LIB_VERSION
ENV SPARK_OPTS --driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info

# Install useful Python packages

# RUN apt-get install -y libxrender1 && apt-get clean
RUN apt-get -y update && \
	apt-get install -y libxrender1 fonts-dejavu && \
	apt-get clean
#RUN conda create --yes -q -n python3.5-env python=3.5 nose numpy pandas scikit-learn scikit-image matplotlib scipy seaborn sympy cython patsy statsmodels cloudpickle numba bokeh pillow ipython jsonschema boto
RUN conda install --yes nose numpy pandas scikit-learn scikit-image matplotlib scipy seaborn sympy cython patsy statsmodels cloudpickle numba bokeh pillow ipython jsonschema boto

ENV PATH $CONDA_DIR/bin:$PATH
RUN conda install --yes numpy pandas scikit-learn scikit-image matplotlib scipy seaborn sympy cython patsy statsmodels cloudpickle numba bokeh pillow && conda clean -yt
RUN /bin/bash -c "pip install mistune"

# Bolt setup
# use the latest from the master branch
RUN git clone https://github.com/bolt-project/bolt
RUN /bin/bash -c "pip install -r bolt/requirements.txt"
ENV BOLT_ROOT $HOME/bolt
ENV PATH $PATH:$BOLT_ROOT/bin
ENV PYTHONPATH $PYTHONPATH:$BOLT_ROOT

USER pyqae_user

# Install Python 3 Tensorflow 0.10.0
RUN conda install --quiet --yes --channel https://conda.anaconda.org/conda-forge 'tensorflow=0.10.0'
# Install the latest version of Theano
RUN pip install --upgrade git+git://github.com/Theano/Theano.git
# Keras
RUN conda install --channel https://conda.anaconda.org/KEHANG --quiet --yes 'keras=1.0.8'
# Newest Version / Also OS X compatibility
RUN pip --no-cache-dir install 'keras==1.1.2'

# Simple ITK
RUN conda install --channel https://conda.anaconda.org/SimpleITK --quiet --yes 'SimpleITK=0.10.0'

# Nibabel
RUN conda install --channel https://conda.anaconda.org/conda-forge --quiet --yes 'nibabel=2.1.0'


# GDAL
# RUN conda install --channel https://conda.anaconda.org/conda-forge --quiet --yes 'gdal=2.1.2'
# RUN pip install -i https://pypi.anaconda.org/pypi/simple 'pygdal==1.11.0.0'

# Install Jupyter notebook as pyqae_user
RUN conda install --quiet --yes \
    'notebook=4.2*' \
    && conda clean -tipsy

# Install JupyterHub to get the jupyterhub-singleuser startup script

RUN pip --no-cache-dir install 'jupyterhub==0.5'

# setup shared folders
USER root

# test data
ADD data $HOME/data

# spark extensions / packages
ADD jars $HOME/spark_jars

# Add any prebuilt python packages in the eggs folder
ADD pyqae $HOME/pyqae
ADD setup.py $HOME/setup.py
ADD *.txt $HOME

# Setup for standard runtime
# Add the notebooks directory
ADD notebooks $HOME/notebooks

# Set permissions on the notebooks
RUN chown -R pyqae_user:pyqae_user $HOME/notebooks
RUN chown -R pyqae_user:pyqae_user $HOME/data
RUN chown -R pyqae_user:pyqae_user $HOME/spark_jars

#TODO FIX RUN easy_install $HOME/py_eggs/*.egg

WORKDIR $HOME
RUN python setup.py install

# Switch back to non-root user
USER pyqae_user

WORKDIR $HOME/notebooks

# Setup Spark + IPython env vars
ENV PYSPARK_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/jupyter
ENV PYSPARK_DRIVER_PYTHON_OPTS "notebook --ip=0.0.0.0"

CMD /bin/bash -c pyspark