#!/usr/bin/env python
# coding: utf-8

# http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing


from sklearn import preprocessing
import numpy as np


X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

print X
print 'mean:               {}'.format(X.mean(axis=0))
print 'standard deviation: {}'.format(X.std(axis=0))


# Standardize
print
print '---- Standardize(scale) ----'
print

X1 = preprocessing.scale(X)
print X1
print 'mean:               {}'.format(X1.mean(axis=0))
print 'standard deviation: {}'.format(X1.std(axis=0))

print
print '---- Standardize(StandardScaler) ----'
print
scaler = preprocessing.StandardScaler().fit(X)
X2 = scaler.transform(X)
print X2
print 'mean:               {}'.format(X2.mean(axis=0))
print 'standard deviation: {}'.format(X2.std(axis=0))
