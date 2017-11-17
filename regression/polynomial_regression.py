#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy
import matplotlib.pyplot as pyplot

quadratic_featurizer = PolynomialFeatures(degree=2)

# training data

# diameters
x_train = numpy.array([[6], [8], [10], [14], [18]])
# prices
y_train = numpy.array([[7], [9], [13], [17.5], [18]])

x_train_quadratic = quadratic_featurizer.fit_transform(x_train)

# test data

# diameters
x_test = numpy.array([[9], [11], [16], [12]])
# prices
y_test = numpy.array([[8.5], [15], [18], [11]])

x_test_quadratic = quadratic_featurizer.fit_transform(x_test)

# build model

regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)

# predict

predictions = regressor.predict(x_test)
predictions_quadratic = regressor_quadratic.predict(x_test_quadratic)

for i in xrange(len(x_test)):
    print 'when diameter=%.2f, LinearRegression predicted price=%.2f, actual price=%.2f' \
          % (x_test[i][0], predictions[i], y_test[i][0])
    print 'when diameter=%.2f, PolynomialRegression predicted price=%.2f, actual price=%.2f' \
        % (x_test[i][0], predictions_quadratic[i], y_test[i][0])

# score

print 'score of LinearRegression is %.2f' % (regressor.score(x_test, y_test))

print 'score of PolynomialRegression is %.2f' % \
      (regressor_quadratic.score(x_test_quadratic, y_test))

# draw

x = numpy.linspace(0, 30, 10)
y = regressor.predict(x.reshape(x.shape[0], 1))

x_quadratic = quadratic_featurizer.fit_transform(x.reshape(x.shape[0], 1))
y1 = regressor_quadratic.predict(x_quadratic)

pyplot.plot(x, y)
pyplot.plot(x, y1, c='r', linestyle='--')
pyplot.axis([0, 30, 0, 30])
pyplot.grid(True)
pyplot.scatter(x_train, y_train)
pyplot.scatter(x_test, y_test)
pyplot.show()
