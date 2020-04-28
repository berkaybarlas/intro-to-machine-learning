import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

import math

n = 1000;
mu, variance = 0, 0.25

sigma = math.sqrt(variance)
s = np.random.normal(mu, sigma, n)
x = np.linspace(0, 1, n)
y = x + stats.norm.pdf(x, mu, sigma)

#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
# 4.1 Plot yi = xi + i  distribution curve
#plt.plot(x, y)
#plt.show()

# 4.2 Compute min value with respect to a 
def func(a):
     """The function"""
     return sum((a*x[:]-y[:])**2.0)

x0 = 1#np.array([1.3, 0.7, 0.8, 1.9, 1.2, 0])
res = minimize(func, x0,  method='BFGS')

print((np.mean(res.x)))
# 4.3


# 4.4
#fig, ax =plt.ssubplots(1,2)
variance2 = 0.01

sigma2 = math.sqrt(variance2)
y2 = (30*(x - 0.25)**2) * (x - 0.75)**2 + stats.norm.pdf(x, mu, sigma2)
plt.plot(x, y2)
plt.show()