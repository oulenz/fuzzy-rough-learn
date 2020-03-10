"""
====================================================
Fuzzy Rough Nearest Neighbours (FRNN) Classification
====================================================

Sample usage of FRNN classification.
It will plot the decision boundaries for each class.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from frlearn.base import discretise_scores
from frlearn.ensembles import FRNN
from frlearn.neighbours import KDTree
from frlearn.utils.owa_operators import additive, strict

n_neighbors = 15

# import example data and reduce to 2 dimensions
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


for owa_weights, k in [(strict(), 1), (additive(), 20)]:
    # create an instance of the FRNN Classifier and construct the model.
    nn_search = KDTree()
    clf = FRNN(nn_search=nn_search, upper_weights=owa_weights, lower_weights=owa_weights, upper_k=k)
    model = clf.construct(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.query(np.c_[xx.ravel(), yy.ravel()])
    Z = discretise_scores(Z, labels=model.classes)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (weights = {})'.format(owa_weights))

plt.show()

