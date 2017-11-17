#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.linear_model import LinearRegression
import numpy

# training data

# diameters, toppings
x_train = numpy.array([[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]])
# prices
y_train = numpy.array([[7], [9], [13], [17.5], [18]])

# build model

model = LinearRegression()
model.fit(x_train, y_train)

# test data

# diameters, toppings
x_test = numpy.array([[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]])
# prices
y_test = numpy.array([[11], [8.5], [15], [18], [11]])

# predict

predictions = model.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'when diameter=%.2f toppings=%.2f, predicted price=%.2f, actual price=%.2f' \
          % (x_test[i][0], x_test[i][1], prediction, y_test[i][0])

# score

print 'R-squared is %.2f' % (model.score(x_test, y_test))
