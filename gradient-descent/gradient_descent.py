import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/data1.txt', delimiter=',')
data = np.column_stack((np.ones(data.shape[0]), data))
features = data[:, :2]
labels = data[:, 2]
theta = np.array([0.0, 0.0])

# SINGLE VARIABLE LINEAR REGRESSION #

# Calculates the cost according to the LMS
# Input: 
# - np.array of size (M x 2)
# - np.array of size (M x 1)
# - np.array of size (2)
# Output:
# - scalar cost
def cost(X, y, theta):
    m = y.shape[0]
    J = 0
    for i in range(m):
        J += (y[i] - X[i].dot(theta))**2
    J /= (2*m)
    return J

# Perform gradient descent on the given dataset
# Input:
# - np.array of size (M x 2)
# - np.array of size (M x 1)
# - np.array of size (2)
# - float alpha (learning rate)
# - integer number of iterations to perforom
# Output:
# - theta parameters
# - cost history
def gradientDescent(X, y, theta, alpha, num_iterations):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    
    for i in range(num_iterations):
        batch = [0.0, 0.0]
        for j in range(m):
            batch[0] += (X[j].dot(theta) - y[j])*X[j][0]
            batch[1] += (X[j].dot(theta) - y[j])*X[j][1]
        batch[0], batch[1] = batch[0] * (alpha / m), batch[1] * (alpha / m)
        theta -= batch
        J_history.append(cost(X, y, theta))
    return theta, J_history
    
# Make a prediction using the linear regression
# Input:
# - number (input data)
# Output:
# - number (prediction)
def predict(x, theta):
    return theta[0] + x*theta[1]
    
theta, y = gradientDescent(features, labels, theta, 0.01, 1000)
plt.figure()
plt.plot(np.arange(len(y)), y)
plt.title("Iterations vs. Cost")
plt.ylabel('Cost')
plt.xlabel('Iterations')

plt.figure()
plt.plot(features[:, 1], labels, 'ro', ms='10', mec="0")
plt.plot(features[:, 1], features.dot(theta), '-')
plt.title("City Population vs. Food Truck Profit")
plt.ylabel("Profit ($10,000)")
plt.xlabel("Population (10,000)")

    