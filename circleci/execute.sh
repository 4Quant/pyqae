export PATH="$HOME/miniconda3/bin:$PATH"
source activate pyqae
export PYQAE_HOME=$(pwd)
# add pyspark code
SPARK_HOME=$HOME/spark-2.3.0-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_SUBMIT_ARGS="--master local[*] pyspark-shell"

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

. ${PYQAE_HOME}/run-tests.sh

# Generating notebooks
for nb in notebooks/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=3600 --execute "$nb" --to markdown |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
done

cd ~
mkdir -p ${PYQAE_HOME}/doc
mkdir -p ${PYQAE_HOME}/doc/notebooks
mkdir -p ${PYQAE_HOME}/doc/Pipelines
cp -r ${PYQAE_HOME}/notebooks/Pipelines/* ${PYQAE_HOME}/doc/Pipelines
cp -r ./doc ${CIRCLE_ARTIFACTS}
