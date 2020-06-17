#####################
fuzzy-rough-learn API
#####################

This is the full API documentation of `fuzzy-rough-learn`.


Neighbours
==========

.. automodule:: frlearn.neighbours
    :no-members:
    :no-inherited-members:

.. currentmodule:: frlearn

Classifiers
-----------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbours.classifiers.FRNN
   neighbours.classifiers.FROVOCO
   neighbours.classifiers.FRONEC

Nearest Neighbour Search
------------------------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbours.neighbour_search.NNSearch
   neighbours.neighbour_search.BallTree
   neighbours.neighbour_search.KDTree

Preprocessors
-------------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbours.preprocessors.FRFS
   neighbours.preprocessors.FRPS

Utils
=====

.. automodule:: frlearn.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: frlearn

numpy utils
-------------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.np_utils.first
   utils.np_utils.last
   utils.np_utils.least
   utils.np_utils.greatest
   utils.np_utils.div_or

OWA Operators
-------------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   utils.owa_operators.OWAOperator
