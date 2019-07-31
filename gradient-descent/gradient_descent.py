import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculates the cost according to the LMS
# Input: 
# - np.array of size (M x N) containing features
# - np.array of size (M x 1) containing labels
# - np.array of size (N) containing theta parameters
# Output:
# - scalar cost
def cost(X, y, theta):
    return (1 / (2*y.shape[0])) * np.sum(np.square(X.dot(theta) - y))

# Normalize a dataset
# Input:
# - np.array of size (M x N) containing features
# Output:
# - np.array of size (M x N) containing normalized features
# - np.array of size (N) containing feature means
# - np.array of size (N) contains feature standard deviations
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    for feature_index in range(X.shape[1]):
        mean = np.mean(X_norm[:, feature_index])
        std = np.std(X_norm[:, feature_index])
        X_norm[:, feature_index] -= mean
        X_norm[:, feature_index] /= std
        mu[feature_index] = mean
        sigma[feature_index] = std
    return X_norm, mu, sigma


# SINGLE VARIABLE LINEAR REGRESSION #

# Perform gradient descent on the given dataset
# Input:
# - np.array of size (M x 2) containing features
# - np.array of size (M x 1) containing labels
# - np.array of size (2) containing theta parameters
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
# - number representing input variable
# Output:
# - number representing the model's prediction
def predict(x, theta):
    return theta[0] + x*theta[1]
    
    
# MULTI VARIABLE LINEAR REGRESSION #
    
# Perform gradient descent on the given dataset
# Input:
# - np.array of size (M x N) containing features
# - np.array of size (M x 1) containing labels
# - np.array of size (N) containing theta parameters
# - float alpha (learning rate)
# - integer number of iterations to perforom
# Output:
# - np.array containing theta parameters
# - list containing cost histories
def gradientDescentMultivar(X, y, theta, alpha, num_iterations):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    
    for i in range(num_iterations):
        batch = np.zeros(theta.shape[0])    
        for j in range(m):
            for feature_index in range (X.shape[1]):
                batch[feature_index] += (X[j].dot(theta) - y[j])* X[j][feature_index]
        batch *= (alpha / m)
        theta -= batch
        print(theta)
        J_history.append(cost(X, y, theta))
    return theta, J_history

# Make a prediction using the linear regression
# Input:
# - np.array of size (1 x N) containing a single set of features
# - np.array of size (N) containing theta parameters
# Output:
# - number representing the model's prediction
def predictMultivar(X, theta):
    X = X.copy()
    X = np.insert(X, 0, 1)
    return X.dot(theta)

def demoSinglevar():
    data = np.loadtxt('data/data1.txt', delimiter=',')
    data = np.column_stack((np.ones(data.shape[0]), data))
    features = data[:, :2]
    labels = data[:, 2]
    theta = np.zeros(features.shape[1])
    
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
        
def demoMultivar():    
    data = np.loadtxt('data/data2.txt', delimiter=',')
    data, mu, sigma = featureNormalize(data)
    data = np.column_stack((np.ones(data.shape[0]), data))
    features = data[:, :3]
    labels = data[:, 3]
    theta = np.zeros(features.shape[1])
    theta, J_history = gradientDescentMultivar(features, labels, theta, 0.05, 250)
    print(f'Lowest cost: {J_history[-1]}')
    
    plt.figure()
    plt.plot(np.arange(len(J_history)), J_history)
    plt.title("Iterations vs. Cost")
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    
    example = np.array([2500, 3])
    example = (example - mu[:2]) / sigma[:2]
    prediction = predictMultivar(example, theta) 
    prediction = (prediction * sigma[2]) + mu[2]
    print(prediction)
    
demoMultivar()








    