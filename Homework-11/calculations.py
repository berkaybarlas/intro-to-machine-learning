import numpy as np
import matplotlib.pyplot as plt

yalefaces= np.loadtxt('yalefaces.csv', delimiter=',')

covarience = np.cov(yalefaces)
eigenVal, eigenVec = np.linalg.eig(covarience)
foundTotal = False
variances = []

sumOfVariances = 0 

# Find Variances
for val in eigenVal:
    var = val / np.sum(eigenVal)
    variances.append(var)
# Find Principal Components 
for i in range(len(variances)):
    sumOfVariances += variances[i]
    if sumOfVariances >= 0.95 and not foundTotal:
        print(*[ i, 'principal components are needed for representing %95 of the total variation'])
        foundTotal = True
    if sumOfVariances >= 0.99:
        print(*[ i, 'principal components are needed for representing %99 of the total variation'])
        break

fig = plt.figure()
X = np.linspace(0,len(eigenVal),len(eigenVal))
fig.suptitle('Eigenvalues')
plt.scatter(X, eigenVal, color='blue')
plt.show()

fig = plt.figure()
fig.suptitle('Eigenvectors')
for k in range(0,20):
    plt.subplot(4, 5, k+1)
    plt.imshow(eigenVec[:,k].reshape((48,42)),cmap='gray')
plt.show()
