import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


moon_x = np.loadtxt('./moons/moons.x.csv', delimiter=',')
moon_y = np.loadtxt('./moons/moons.y.csv', delimiter=',')

print("moonx", moon_x)
print("moony", moon_y)
## Describe how to learn the best decision stump efficiently, and give the computational 2
# complexity of your weak learner. 
# Include a plot of the decision region of the optimal decision stump for the moons dataset when the distribution is uniform.

# input: dataset X and labels y (in {+1, -1})
hypotheses = []
hypothesis_weights = []

N, _ = X.shape
d = np.ones(N) / N

for t in range(num_iterations):
    h = DecisionTreeClassifier(max_depth=1)

    h.fit(X, y, sample_weight=d)
    pred = h.predict(X)

    eps = d.dot(pred != y)
    alpha = (np.log(1 - eps) - np.log(eps)) / 2

    d = d * np.exp(- alpha * y * pred)
    d = d / d.sum()

    hypotheses.append(h)
    hypothesis_weights.append(alpha)

 # X input, y output
y = np.zeros(N)
for (h, alpha) in zip(hypotheses, hypothesis_weights):
    y = y + alpha * h.predict(X)
y = np.sign(y)
