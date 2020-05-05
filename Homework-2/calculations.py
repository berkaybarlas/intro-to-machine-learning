import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

import math

## Formatting
plt.rcParams.keys()

plt.rcParams['font.size'] = 7
#plt.rcParams['lines.makersize'] = 4
plt.rcParams['figure.figsize'] = [5, 3]
plt.rcParams['figure.dpi'] = 150



mu, variance = 0, 0.25
sigma = math.sqrt(variance)
w0 = 1
w1 = 1

## 1.5 Simulate part 4
nValues = np.arange(1, 100, 1)
D = 4
errorsN = []
for n in nValues:
     a = np.linspace(-1, 1, n)
     z = np.random.normal(mu, sigma, n)
     ys = w1 * a + w0
     y = ys +  z

     # degrees
     deg = np.arange(D)
     # transpose 1 dimension array
     aT = np.transpose(a[np.newaxis])
     X = aT ** deg
     XT = np.transpose(X)
     pseduoInv = np.linalg.pinv(XT.dot(X))
     wCap = pseduoInv.dot(XT).dot(y)

     XwCap = X.dot(wCap)
     error = XwCap-ys
     averageE = np.linalg.norm(error, 2)**2
     errorsN.append(averageE/n)

dValues = np.arange(2, 100, 1)
n = 150
errorsD = []
for D in dValues:
     a = np.linspace(-1, 1, n)
     z = np.random.normal(mu, sigma, n)
     ys = w1 * a + w0
     y = ys +  z

     # degrees
     deg = np.arange(D)
     # transpose 1 dimension array
     aT = np.transpose(a[np.newaxis])
     X = aT ** deg
     XT = np.transpose(X)
     pseduoInv = np.linalg.pinv(XT.dot(X))
     wCap = pseduoInv.dot(XT).dot(y)

     XwCap = X.dot(wCap)
     error = XwCap-ys
     averageE = np.linalg.norm(error, 2)**2
     errorsD.append(averageE/n)


#plt.scatter(nValues, errorsN, label="function of n")
plt.scatter(dValues, errorsD, color= "red", label="function of D")
plt.xlabel("value of d")
plt.ylabel("Average Error")
plt.title("Part 5")
plt.legend()
plt.show()

## 1.7 Simulate part 6
a1 = np.linspace(-4, 3, n)
s = np.random.normal(mu, sigma, n)
# y = x + stats.norm.pdf(x, mu, sigma)

# Average over multiple realizations of the noise

#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
## 4.1 Plot yi = xi + i  distribution curve