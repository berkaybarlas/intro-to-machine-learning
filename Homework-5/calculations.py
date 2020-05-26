from matplotlib import pyplot as plt
import numpy as np
import math

x = 0
x_star = 5.25
a_k = 1

steps = np.linspace(0,99,100);
errors = []
for i in steps:
     #a_k = (5/6)**i
     a_k = 1 / (i+1)
     gradient = 1 / 2 * (4*x - 21) / math.sqrt(2*(x**2) - 21 * x + 56.25)
     x_next = x - a_k * gradient
     rate = (x_star - x_next) / x_star
     rate = abs(rate)

     if( rate <= 0.01):
          print("found", rate, x_next,x, i)
     errors.append(rate)
     x = x_next

plt.scatter(steps, errors, color= "red")
plt.xlabel("Steps")
plt.ylabel("Error")
plt.title("Problem 1")
plt.show()

x = 0
a_k = 0.1
errors2 = []
#steps = np.linspace(0,99,100);
steps = np.linspace(0,10,11);
for i in steps:
     #a_k = (1/6)**i
     a_k = 1 / (4 *(i+1))
     gradient = 4 * x - 21
     x_next = x - a_k * gradient
     rate = (x_star - x_next) / x_star
     rate = abs(rate)
     if( rate <= 0.01):
          print("found2", rate, x_next,x, i)
     errors2.append(rate)
     x = x_next

plt.scatter(steps, errors2, color= "blue")
plt.xlabel("Steps")
plt.ylabel("Error")
plt.title("Problem 1 ")
plt.show()