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

**fuzzy-rough-learn** is a library of machine learning algorithms involving fuzzy rough sets, as well as data descriptors that can be used for one-class classification / novelty detection. It builds on scikit-learn_, but uses a slightly different api.


Contents
--------

At present, fuzzy-rough-learn contains the following algorithms:

Classifiers
...........

* Fuzzy Rough Nearest Neighbours (FRNN; multiclass)
* Fuzzy Rough OVO COmbination (FROVOCO; muliclass, suitable for imbalanced data)
* Fuzzy ROugh NEighbourhood Consensus (FRONEC; multilabel)

Data descriptors
................

* Average Localised Proximity (ALP)
* Centre Distance (CD)
* Extended Isolation Forest (EIF)
* Isolation Forest (IF)
* Local Outlier Factor (LOF)
* Localised Nearest Neighbour Distance (LNND)
* Mahalanobis Distance (MD)
* Nearest Neighbour Distance (NND)
* Support Vector Machine (SVM)

Preprocessors
.............

* Fuzzy Rough Feature Selection (FRFS)
* Fuzzy Rough Prototype Selection (FRPS)
* Shrink Autoencoder (SAE; unsupervised)

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

In addition, some algorithms require optional dependencies:

* eif >= 2.0.0 (EIF)
* tensorflow >= 2.2.0 (SAE)

Citing fuzzy-rough-learn
------------------------

If you use or refer to fuzzy-rough-learn in a scientific publication, please cite `this paper <https://link.springer.com/chapter/10.1007%2F978-3-030-52705-1_36>`_::

  Lenz OU, Peralta D, Cornelis C (2020).
  fuzzy-rough-learn 0.1: a Python library for machine learning with fuzzy rough sets.
  IJCRS 2020: Proceedings of the International Joint Conference on Rough Sets, pp 491–499.
  Lecture Notes in Artificial Intelligence, vol 12179, Springer.
  doi: 10.1007/978-3-030-52705-1_36

Bibtex entry::

  @inproceedings{lenz20fuzzyroughlearn,
    title={{f}uzzy-rough-learn 0.1: a {P}ython library for machine learning with fuzzy rough sets},
    author={Lenz, Oliver Urs and Peralta, Daniel and Cornelis, Chris},
    booktitle={{IJCRS} 2020: Proceedings of the International Joint Conference on Rough Sets},
    pages={491--499},
    year={2020},
    series={Lecture Notes in Artificial Intelligence},
    volume={12179},
    publisher={Springer}
  }
