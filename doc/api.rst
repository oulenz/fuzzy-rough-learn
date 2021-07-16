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
   :template: class.rst

   FRNN
   FROVOCO
   FRONEC

Data descriptors
================

.. currentmodule:: frlearn.data_descriptors

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

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
   :template: class.rst

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
   :template: class.rst

   FRFS
   NormNormaliser

Instance preprocessors
======================

.. currentmodule:: frlearn.instance_preprocessors

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   FRPS

Other
=====

Dispersion measures
-------------------

.. currentmodule:: frlearn.dispersion_measures

.. autosummary::
   :toctree: generated/
   :nosignatures:

    interquartile_range
    maximum_absolute_value
    standard_deviation
    total_range

Location measures
-----------------

.. currentmodule:: frlearn.location_measures

.. autosummary::
   :toctree: generated/
   :nosignatures:

   maximum
   mean
   median
   midhinge
   midrange
   minimum


Nearest neighbour search algorithms
-----------------------------------

.. currentmodule:: frlearn.neighbour_search_methods

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   BallTree
   KDTree

Numpy
-----

.. currentmodule:: frlearn.numpy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   div_or
   first
   greatest
   last
   least
   remove_diagonal
   soft_head
   soft_max
   soft_min
   soft_tail

Parametrisations
----------------

.. currentmodule:: frlearn.parametrisations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fraction
   log_units

T-norms
-------

.. currentmodule:: frlearn.t_norms

.. autosummary::
   :toctree: generated/
   :nosignatures:

   goguen_t_norm
   heyting_t_norm
   lukasiewicz_t_norm

Transformations
---------------

.. currentmodule:: frlearn.transformations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   contract
   shifted_reciprocal
   truncated_complement

Weights
-------

.. currentmodule:: frlearn.weights

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   Weights
   ConstantWeights
   ExponentialWeights
   LinearWeights
   QuantifierWeights
   ReciprocallyLinearWeights
