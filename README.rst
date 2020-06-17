.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/oulenz/fuzzy-rough-learn.svg?branch=master
.. _Travis: https://travis-ci.com/oulenz/fuzzy-rough-learn

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/7xrrtwcj0i3lgd5a/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/oulenz/fuzzy-rough-learn

.. |Codecov| image:: https://codecov.io/gh/oulenz/fuzzy-rough-learn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/oulenz/fuzzy-rough-learn

.. |CircleCI| image:: https://circleci.com/gh/oulenz/fuzzy-rough-learn.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/oulenz/fuzzy-rough-learn/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/fuzzy-rough-learn/badge/?version=latest
.. _ReadTheDocs: https://fuzzy-rough-learn.readthedocs.io/en/latest/?badge=latest

.. _scikit-learn: https://scikit-learn.org

fuzzy-rough-learn
=================

**fuzzy-rough-learn** is a library of fuzzy rough machine learning algorithms, extending scikit-learn_.


Contents
--------

At present, fuzzy-rough-learn contains the following algorithms:

Classifiers
...........

* Fuzzy Rough Nearest Neighbours (FRNN; multiclass)
* Fuzzy Rough OVO COmbination (FROVOCO; muliclass, suitable for imbalanced data)
* Fuzzy ROugh NEighbourhood Consensus (FRONEC; multilabel)

Preprocessors
.............

* Fuzzy Rough Feature Selection (FRFS)
* Fuzzy Rough Prototype Selection (FRPS)

Utilities
.........

* OWA operator class
* Nearest Neighbour search algorithm class


Documentation
-------------

The documentation is located here_.

.. _here: https://fuzzy-rough-learn.readthedocs.io/en/stable/


Dependencies
------------

fuzzy-rough-learn requires python 3.7+ and the following packages:

* scipy >= 1.1.0
* numpy >=1.16.0
* scikit-learn >=0.22.0
