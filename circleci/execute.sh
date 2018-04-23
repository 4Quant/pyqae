export PATH="$HOME/miniconda3/bin:$PATH"
source activate pyqae
export PYQAE_HOME=$(pwd)

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Generating lectures
for nb in notebooks/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=3600 --execute "$nb" --to markdown |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
done

. ${PYQAE_HOME}/run-tests.sh

cd ~
mkdir -p ${PYQAE_HOME}/doc
mkdir -p ${PYQAE_HOME}/doc/notebooks
cp -r ${PYQAE_HOME}/notebooks/* ${PYQAE_HOME}/doc/notebooks
cp -r ./doc ${CIRCLE_ARTIFACTS}
