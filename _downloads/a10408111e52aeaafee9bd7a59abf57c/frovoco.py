"""
=================================================
Imbalanced multiclass classification with FROVOCO
=================================================

The figures contain the training instances within a section of the selected feature space.
The training instances are coloured according to their true labels,
while the feature space is coloured according to predictions on the basis of the training instances,
making the decision boundaries visible.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from frlearn.base import select_class
from frlearn.classifiers import FROVOCO

# Import example data, reduce to 2 dimensions, and create imbalanced selection.
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X = np.concatenate([X[:5], X[50:67], X[100:]], axis=0)
y = np.concatenate([y[:5], y[50:67], y[100:]], axis=0)

# Define color maps.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create an instance of the FROVOCO classifier and construct the model.
clf = FROVOCO()
model = clf(X, y)

# Create a mesh of points in the attribute space.
step_size = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Query mesh points to obtain class values and select highest valued class.
Z = model(np.c_[xx.ravel(), yy.ravel()])
Z = select_class(Z, labels=model.classes)

# Initialise figure.
plt.figure()

# Plot mesh.
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot training instances.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)

# Set plot dimensions.
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('FROVOCO applied to an imbalanced selection of iris dataset')

plt.show()

