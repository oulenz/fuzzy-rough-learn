"""
============================
Instance selection with FRPS
============================

Sample usage of FRPS preprocessing, demonstrated in combination with (strict) FRNN classification.

The figures contain the selected prototypes within a section of the feature space.
The prototypes are coloured according to their true labels,
while the feature space is coloured according to predictions on the basis of the prototypes,
making the decision boundaries visible.

In total, nine subfigures are displayed,
to illustrate the effect of the `quality_measure` (rows) and `aggr_R` (columns) parameters.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from frlearn.base import select_class
from frlearn.classifiers import FRNN
from frlearn.instance_preprocessors import FRPS
from frlearn.t_norms import heyting_t_norm, lukasiewicz_t_norm

# Import example data and reduce to 2 dimensions.
iris = datasets.load_iris()
X_orig = iris.data[:, :2]
y_orig = iris.target

# Create a mesh of points in the attribute space.
step_size = .02
x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Define color maps.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Initialise figure.
plt.figure()

for i, (aggr_name, aggr_R) in enumerate([('mean', np.mean), ('Łukasiewicz', lukasiewicz_t_norm), ('Heyting', heyting_t_norm)]):
    for j, quality_measure in enumerate(['upper', 'lower', 'both']):
        axes = plt.subplot(3, 3, i*3 + j + 1)

        # Create an instance of the FRPS preprocessor and process the data.
        preprocessor = FRPS(aggr_R=aggr_R, quality_measure=quality_measure)
        X, y = preprocessor(X_orig, y_orig)

        # Create an instance of the FRNN classifier and construct the model.
        clf = FRNN(upper_weights=None, lower_weights=None, upper_k=1, lower_k=1)
        model = clf(X, y)

        # Query mesh points to obtain class values and select highest valued class.
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = select_class(Z, labels=model.classes)

        # Plot mesh.
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot training instances.
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

        # Set plot dimensions.
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Describe columns and rows.
        if axes.is_first_col():
            plt.ylabel(aggr_name, rotation=0, size='large', ha='right')
        if axes.is_first_row():
            plt.title(quality_measure)

plt.suptitle('FRNN applied to instances of iris dataset selected by FRPS', fontsize=14)
plt.show()
