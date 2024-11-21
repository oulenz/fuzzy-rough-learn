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

fuzzy-rough-learn
=================

**fuzzy-rough-learn** is a library of machine learning algorithms involving fuzzy rough sets, as well as data descriptors that can be used for one-class classification / novelty detection. It builds on scikit-learn_, but uses a slightly different api, best illustrated with a concrete example::

    from sklearn import datasets
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from frlearn.base import probabilities_from_scores, select_class
    from frlearn.classifiers import FRNN
    from frlearn.feature_preprocessors import RangeNormaliser

    # Import example data.
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    # Create an instance of the FRNN classifier, construct the model, and query on the test set.
    clf = FRNN(preprocessors=(RangeNormaliser(), ))
    model = clf(X_train, y_train)
    scores = model(X_test)

    # Convert scores to probabilities and calculate the AUROC.
    probabilities = probabilities_from_scores(scores)
    auroc = roc_auc_score(y_test, probabilities, multi_class='ovo')
    print('AUROC:', auroc)

    # Select classes with the highest scores and calculate the accuracy.
    classes = select_class(scores)
    accuracy = accuracy_score(y_test, classes)
    print('accuracy:', accuracy)

Both classifiers and feature preprocessors are functions that take training data and output a model. Models are functions that take data and output something else. Classifier models output class scores, preprocessor models output a transformation of the data. Preprocessors can be added as a keyword argument when initialising a classifier, which automatically creates a preprocessor model on the basis of the training data and applies it to the training and the test data.

.. _scikit-learn: https://scikit-learn.org


Contents
--------

At present, fuzzy-rough-learn contains the following algorithms:

Multiclass classifiers
......................

* Fuzzy Rough Nearest Neighbours (FRNN; multiclass)
* Fuzzy Rough OVO COmbination (FROVOCO; muliclass, suitable for imbalanced data)
* Fuzzy ROugh NEighbourhood Consensus (FRONEC; multilabel)
* Nearest Neighbours (NN; multiclass)

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

Regressors
..........

* Fuzzy Rough Nearest Neighbours (FRNN)

Feature preprocessors
.....................

* Fuzzy Rough Feature Selection (FRFS)
* Linear normaliser; in particular:

    * Interquartile range (IQR) normaliser
    * Maximum absolute value (MaxAbs) normaliser
    * Range normaliser
    * Standardiser

* Shrink Autoencoder (SAE; unsupervised)
* Vector size normaliser

Instance preprocessors
......................

* Fuzzy Rough Prototype Selection (FRPS)

Other
.....

* array functions
* dispersion measures
* location measures
* nearest neighbour search methods
* parametrisations
* t_norms
* transformations
* vector size measures
* weights


Documentation
-------------

The documentation is located here_.

.. _here: https://fuzzy-rough-learn.readthedocs.io/en/stable/


Dependencies
------------

fuzzy-rough-learn requires python 3.7+ and the following packages:

* scipy >= 1.1.0
* numpy >=1.17.0
* scikit-learn >=0.24.0

In addition, some algorithms require optional dependencies:

* eif >= 2.0.0 (EIF)
* tensorflow >= 2.2.0 (SAE)

Citing fuzzy-rough-learn
------------------------

If you use or refer to fuzzy-rough-learn in a scientific publication, please cite `this paper <https://ieeexplore.ieee.org/document/9882778>`_::

  Lenz OU, Cornelis C, Peralta D (2022).
  fuzzy-rough-learn 0.2: a Python library for fuzzy rough set algorithms and one-class classification.
  FUZZ-IEEE 2022: Proceedings of the IEEE International Conference on Fuzzy Systems.
  doi: 10.1109/FUZZ-IEEE55066.2022.9882778

Bibtex entry::

  @inproceedings{lenz22fuzzyroughlearn,
    title={{f}uzzy-rough-learn 0.2: a {P}ython library for fuzzy rough set algorithms and one-class classification},
    author={Lenz, Oliver Urs and Cornelis, Chris and Peralta, Daniel},
    booktitle={{FUZZ-IEEE} 2022: Proceedings of the IEEE International Conference on Fuzzy Systems},
    year={2022},
    publisher={IEEE},
  }
