#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone fuzzy-rough-learn/frlearn repository into a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

make_mamba() {
	TO_INSTALL="$@"
    # Deactivate the travis-provided virtual environment and setup a
    # mamba-based environment instead
    deactivate

    # Install mambaforge
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        -O mambaforge.sh
    MAMBAFORGE_PATH=/home/travis/mambaforge
    chmod +x mambaforge.sh && ./mambaforge.sh -b -p $MAMBAFORGE_PATH
    export PATH=$MAMBAFORGE_PATH/bin:$PATH
    mamba update --yes mamba

    mamba create -n testenv --yes $TO_INSTALL
    source activate testenv
}

TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"
  make_mamba $TO_INSTALL

if [[ "$SKLEARN_VERSION" == "nightly" ]]; then
    mamba install --yes cython
    pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
else
    mamba install --yes scikit-learn=$SKLEARN_VERSION
fi

pip install coverage codecov
pip install sphinx numpydoc  # numpydoc requires sphinx

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
ccache --show-stats
# Useful for debugging how ccache is used
# cat $CCACHE_LOGFILE
