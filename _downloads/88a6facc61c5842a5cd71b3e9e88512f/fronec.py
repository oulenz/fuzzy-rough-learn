"""
=====================================
Multilabel classification with FRONEC
=====================================

The figures contain training instances within a section of the feature space.
The training instances are coloured according to their true labels,
while the feature space is coloured according to predictions on the basis of the training instances,
making the decision boundaries visible.

There are three labels, encoded by the colours red, blue and yellow,
while green means both blue and yellow, and white means no label.

In total, six subfigures are displayed, to illustrate the effect of the `Q_type` and `R_d_type` parameters.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn import datasets

from frlearn.base import discretise
from frlearn.classifiers import FRONEC

# Import example data, reduce to 2 dimensions and manipulate to create multilabel problem.
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
Y = (y[:, None] == np.arange(3)).astype(int)
Y[[109, 117, 131], 2] = 0
Y[((X[:, 0] >= 6) & (y == 1)), 2] = 1
Y[((X[:, 0] <= 6) & (y == 2)), 1] = 1

# Create a mesh of points in the attribute space.
step_size = .1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Define light and dark colour maps for label combinations.
light_colours = ['#FFFFFF', '#FF8080', '#8080FF', '#C080C0', '#FFFF80', '#FFC080', '#80C080', '#C0C0C0']
light_cmap = ListedColormap(light_colours)
dark_colours = ['#FFFFFF', '#FF0000', '#0000FF', '#800080', '#FFFF00', '#FF8000', '#008000', '#808080']
dark_cmap = ListedColormap(dark_colours)
norm = BoundaryNorm(np.arange(9), 8)

# Initialise figure with wide aspect for two side-by-side subfigures.
plt.figure(figsize=(8, 4))

for i, R_d_type in enumerate([1, 2]):
    for j, Q_type in enumerate([1, 2, 3]):
        axes = plt.subplot(2, 3, i*3 + j + 1)

        # Create an instance of the FRONEC classifier and construct the model.
        clf = FRONEC(k=10, Q_type=Q_type, R_d_type=R_d_type)
        model = clf(X, Y)

        # Query mesh points to obtain label values and convert into discrete predictions.
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = discretise(Z)

        # Encode label combinations as unique numbers, reshape into mesh and plot with light colour map.
        Z = np.sum(Z * [1, 2, 4], axis=-1)
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=light_cmap, norm=norm)

        # Encode training instance label combinations as unique numbers and plot.
        C = np.sum(Y * [1, 2, 4], axis=-1)
        plt.scatter(X[:, 0], X[:, 1], c=C, cmap=dark_cmap, norm=norm, edgecolor='k', s=20)

        # Set subplot aspect to standard aspect ratio.
        axes.set_aspect(1.0 / axes.get_data_ratio() * .75)

        # Set plot dimensions.
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Describe columns and rows.
        if axes.is_first_col():
            plt.ylabel('R_d^{}'.format(R_d_type), rotation=0, size='large', ha='right')
        if axes.is_first_row():
            plt.title('Q^{}'.format(Q_type))

plt.tight_layout()
plt.show()
