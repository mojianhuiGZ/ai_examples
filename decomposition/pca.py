#!/usr/bin/env python
# coding: utf-8

# http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


# prepare IRIS data

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target
target_names = iris_data.target_names

print 'IRIS size: {}'.format(X.shape)


# PCA

pca = PCA(n_components=0.99)
X1 = pca.fit(X).transform(X)

print 'IRIS size after PCA: {}, explained variance ratio is {}'.format(
    X1.shape,
    pca.explained_variance_ratio_)
