#####################
fuzzy-rough-learn API
#####################

This is the full API documentation of `fuzzy-rough-learn`.

Classifiers
===========

.. currentmodule:: frlearn.classifiers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FRNN
   FROVOCO
   FRONEC

Data descriptors
================

.. currentmodule:: frlearn.data_descriptors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ALP
   CD
   EIF
   IF
   LNND
   LOF
   MD
   NND
   SVM

Feature preprocessors
=====================

Linear normalisers
------------------

.. currentmodule:: frlearn.feature_preprocessors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinearNormaliser
   IQRNormaliser
   MaxAbsNormaliser
   RangeNormaliser
   Standardiser

Other
-----

.. currentmodule:: frlearn.feature_preprocessors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FRFS

Instance preprocessors
======================

.. currentmodule:: frlearn.instance_preprocessors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FRPS

Other
=====

Nearest neighbour search algorithms
-----------------------------------

.. currentmodule:: frlearn.neighbours.neighbour_search

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   NNSearch
   BallTree
   KDTree

OWA Operators
-------------

.. currentmodule:: frlearn.utils.owa_operators

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   OWAOperator

numpy utils
-----------

.. currentmodule:: frlearn.utils.np_utils

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function.rst

   first
   last
   least
   greatest
   div_or
