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

   neighbours.preprocessors.FRPS

Ensembles
=========

.. automodule:: frlearn.ensembles
    :no-members:
    :no-inherited-members:

.. currentmodule:: frlearn

Classifiers
-----------

.. currentmodule:: frlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensembles.classifiers.FRNN
   ensembles.classifiers.FROVOCO
   ensembles.classifiers.FRONEC

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
