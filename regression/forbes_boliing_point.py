#!/usr/bin/env python
# -*- coding:utf-8 -*-

# http://www.1010jiajiao.com/gzsx/shiti_page_133615

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy
import pandas
import matplotlib.pyplot as pyplot

data = [[194.5, 20.79, 1.3179],
        [194.3, 20.79, 1.3179],
        [197.9, 22.40, 1.3502],
        [198.4, 22.67, 1.3555],
        [199.4, 23.15, 1.3646],
        [199.9, 23.35, 1.3683],
        [200.9, 23.89, 1.3782],
        [201.1, 23.99, 1.3800],
        [201.4, 24.02, 1.3806],
        [201.3, 24.01, 1.3805],
        [203.6, 25.14, 1.4004],
        [204.6, 26.57, 1.4244],
        [209.5, 28.49, 1.4547],
        [208.6, 27.76, 1.4434],
        [210.7, 29.04, 1.4630],
        [211.9, 29.88, 1.4754],
        [212.2, 30.06, 1.4780]]

data_frame = pandas.DataFrame(data, columns=['boiling_point', 'pressure', 'log(pressure)'])

x = data_frame['pressure'].values
y = data_frame['boiling_point'].values
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print 'score: %.2f' % (regressor.score(x_test, y_test))

scores = cross_val_score(regressor, x, y, cv=4)
print 'scores: {}'.format(scores)

x_draw = numpy.linspace(16, 32, 20)
y_draw = regressor.predict(x_draw.reshape(x_draw.shape[0], 1))

pyplot.scatter(x, y)
pyplot.plot(x_draw, y_draw, c='r', linestyle='--')
pyplot.grid(True)
pyplot.show()
