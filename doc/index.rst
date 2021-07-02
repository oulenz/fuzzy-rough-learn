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

If you use or refer to fuzzy-rough-learn in a scientific publication, please cite `this paper <https://link.springer.com/chapter/10.1007%2F978-3-030-52705-1_36>`_:

.. code-block:: text

  Lenz OU, Peralta D, Cornelis C (2020).
  fuzzy-rough-learn 0.1: a Python library for machine learning with fuzzy rough sets.
  IJCRS 2020: Proceedings of the International Joint Conference on Rough Sets, pp 491–499.
  Lecture Notes in Artificial Intelligence, vol 12179, Springer.
  doi: 10.1007/978-3-030-52705-1_36

Bibtex entry:

.. code-block:: text

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
