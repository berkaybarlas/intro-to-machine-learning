import matplotlib.pyplot as plt
import numpy as np

moon_x = np.loadtxt('./moons/moons.x.csv', delimiter=',')
moon_y = np.loadtxt('./moons/moons.y.csv', delimiter=',')

## Describe how to learn the best decision stump efficiently, and give the computational 2
# complexity of your weak learner. 
# Include a plot of the decision region of the optimal decision stump for the moons dataset when the distribution is uniform.

# input: dataset X and labels y (in {+1, -1})

N, _ = moon_x.shape

merged = np.concatenate((moon_x,moon_y[:,None]),axis=1)

train, test = np.split(merged, [int(0.8 * N)])

print("split", len(train), len(test), N)
#print("split", train)

sorted_train = np.sort(train.view('f8,f8,f8'), order=['f0'], axis=0)

best_error = 1000
best_point_x = 0
sorted_train = sorted_train[:,0]
#print("split", sorted_train)
t_N = sorted_train.shape[0]

for i in range(t_N-1):
    # chose split point
    split_point = (sorted_train[i][0] + sorted_train[i+1][0]) / 2
    # copy data
    error = 0
    for k in range(t_N):
        if sorted_train[k][0] < split_point and sorted_train[k][2] == 1:
            error += 1
        if sorted_train[k][0] > split_point and sorted_train[k][2] == -1:
            error += 1
    if best_error > error:
        best_error = error
        best_point_x = split_point
print("best x split ",best_error, best_point_x)

sorted_train = np.sort(train.view('f8,f8,f8'), order=['f1'], axis=0)
best_error = 1000
best_point_y = 0
sorted_train = sorted_train[:,0]

for i in range(t_N-1):
    # chose split point
    split_point = (sorted_train[i][1] + sorted_train[i+1][1]) / 2
    # copy data
    error = 0
    for k in range(t_N):
        if sorted_train[k][1] < split_point and sorted_train[k][2] == -1:
            error += 1
        if sorted_train[k][1] > split_point and sorted_train[k][2] == 1:
            error += 1
    if best_error > error:
        best_error = error
        best_point_y = split_point
print("best y split ",best_error, best_point_y)

# A plot of the decision region of the optimal decision stump for the moons dataset when the distribution is uniform.
plt.scatter(merged[:,0], merged[:,1], c=merged[:,2])
plt.plot([-2,3],[best_point_y,best_point_y], color="black")
plt.plot([best_point_x,best_point_x],[-1.3,1.8], color="black")
plt.show()
