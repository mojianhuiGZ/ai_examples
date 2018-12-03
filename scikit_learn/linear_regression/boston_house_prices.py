# coding: utf-8

from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

boston = datasets.load_boston()
sample_number = len(boston.target)
train_number = (sample_number // 10) * 8
boston_train_X = boston.data[:train_number]
boston_test_X = boston.data[train_number:]
boston_train_y = boston.target[:train_number]
boston_test_y = boston.target[train_number:]

assert len(boston_train_X) + len(boston_test_X) == sample_number
assert len(boston_train_y) + len(boston_test_y) == sample_number

regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
regr.fit(boston_train_X, boston_train_y)
boston_test_y_pred = regr.predict(boston_test_X)

print('coefficients:', regr.coef_)
print('intercept:', regr.intercept_)

print('mean squared error: %.2f'
      % mean_squared_error(boston_test_y, boston_test_y_pred))
print('r2_score: %.2f' % (r2_score(boston_test_y, boston_test_y_pred)))
