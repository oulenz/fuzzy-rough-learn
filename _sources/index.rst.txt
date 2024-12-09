Welcome to the documentation of fuzzy-rough-learn!
==================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   changelog

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

`API Documentation <api.html>`_
-------------------------------

The docstrings of the classes and functions.

`Examples <auto_examples/index.html>`_
--------------------------------------

A series of examples.

`Changelog <changelog.html>`_
------------------------------

Release history of fuzzy-rough-learn.

Citing fuzzy-rough-learn
------------------------

If you use or refer to fuzzy-rough-learn in a scientific publication, please cite `this paper <https://ieeexplore.ieee.org/document/9882778>`_:

.. code-block:: text

  Lenz OU, Cornelis C, Peralta D (2022).
  fuzzy-rough-learn 0.2: a Python library for fuzzy rough set algorithms and one-class classification.
  FUZZ-IEEE 2022: Proceedings of the IEEE International Conference on Fuzzy Systems.
  doi: 10.1109/FUZZ-IEEE55066.2022.9882778

Bibtex entry:

.. code-block:: text

  @inproceedings{lenz22fuzzyroughlearn,
    title={{f}uzzy-rough-learn 0.2: a {P}ython library for fuzzy rough set algorithms and one-class classification},
    author={Lenz, Oliver Urs and Cornelis, Chris and Peralta, Daniel},
    booktitle={{FUZZ-IEEE} 2022: Proceedings of the IEEE International Conference on Fuzzy Systems},
    year={2022},
    publisher={IEEE},
  }
