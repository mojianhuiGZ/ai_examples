#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.linear_model import LinearRegression
import numpy

# training data

# diameters
x_training = numpy.array([[6, 8, 10, 14, 18]]).T
# prices
y_training = numpy.array([[7, 9, 13, 17.5, 18]]).T

# build model

model = LinearRegression()
model.fit(x_training, y_training)

# test data

# diameters
x_test = numpy.array([[8, 9, 11, 16, 12]]).T
# prices
y_test = numpy.array([[11, 8.5, 15, 18, 11]]).T

# score

print 'R-squared is %.4f' % (model.score(x_test, y_test))

# predict

print 'when diameter=%d, price is %.2f' % (12, model.predict(12)[0][0])
