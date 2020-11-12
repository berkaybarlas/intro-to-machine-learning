import csv
import numpy as np
from sklearn import preprocessing 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt
import scipy.optimize as optimization

# load feature variables and their names
x = np.loadtxt("hitters.x.csv", delimiter=",", skiprows=1) 
with open("hitters.x.csv", "r") as f:
     x_colnames = next(csv.reader(f)) 
# load salaries
y = np.loadtxt("hitters.y.csv", delimiter=",", skiprows=1)

n, m = x.shape

# Thats why one feature, which is expressed in a very high magnitude (number), may affect the prediction a lot more than an equally important feature.

# The algorithms which use Euclidean Distance measure are sensitive to Magnitudes. Here feature scaling helps to weigh all the features equally.

# Scale data to std = 1 
x_scaled = preprocessing.scale(x) 
  
# Augment x with bias feature consist of 1s
x_bias = np.ones((len(x),1))
x_augmented = np.hstack((x_bias,x_scaled))

x = x_augmented

print(x_scaled.std(axis=0))

# Create 100 values of λ, evenly spaced in the interval [10−3,107] in log scale

lambda_values = np.logspace(-3,7,num=100)
# Plotting the l2 norm of the regression coefficients versus lambda on a log-log plot.

# Create identity matrix
I = np.identity(m+1)

l2_norm_coeffcients = []
for alpha in lambda_values:
     w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + alpha * I), x.T), y)
     l2_norm_coeffcients.append(np.linalg.norm(w))

plt.plot(l2_norm_coeffcients, lambda_values, c='blue')
plt.yscale('log')
plt.show()

alphaT= 1
# 
w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + alphaT * I), x.T), y)

### plt.plot(x, w*x, c='red')
print(len(w),n,m)
print(len(y))
print(np.linalg.norm(w))

# Least sqaures estimate

#r = np.linalg.matrix_rank(x)
#U, sigma, VT = np.linalg.svd(x, full_matrices=False)
#D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
#V = VT.T
#X_plus = V.dot(D_plus).dot(U.T)
#w_least_square = X_plus.dot(y)
#least_square_error = np.linalg.norm(x.dot(w) - y, ord=2) ** 2

lr = LinearRegression()
lr.fit(x, y)
w_least_square = lr.coef_

ridge = Ridge()

parameters = {'alpha': lambda_values}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)
y_scaled = preprocessing.scale(x) 

ridge_regressor.fit(x, y_scaled)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

ridge = Ridge(alpha=10**-3)
#ridge = Ridge(alpha=ridge_regressor.best_params_["alpha"])
ridge.fit(x, y)

w0 = ridge.coef_

features = np.linspace(1,20,20);
print(features)
print(w0)
plt.scatter(w0, features, label="Ridge regression")
plt.scatter(w_least_square, features, c='red',marker='*', label="Linear Regression")
print(len(x[0]), len(x[:,0]), len(y))
plt.title("Very Small Lambda")
plt.legend()
plt.show()