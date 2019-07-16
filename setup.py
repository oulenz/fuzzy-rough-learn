#! /usr/bin/env python
"""A package of tools for fuzzy rough machine learning."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('frlearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'fuzzy-rough-learn'
DESCRIPTION = 'A package of tools for fuzzy rough machine learning.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Oliver Urs Lenz'
MAINTAINER_EMAIL = 'oliver.urs.lenz@gmail.com'
URL = 'https://github.com/oulenz/fuzzy-rough-learn'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/oulenz/fuzzy-rough-learn'
VERSION = __version__
INSTALL_REQUIRES = ['numpy>=1.15.0', 'scipy>=1.1.0', 'scikit-learn>=0.20.0']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
