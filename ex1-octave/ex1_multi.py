import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# Compute cost function 
def computeCostJ(X, Y, theta):
    y_hat = np.matmul(X, theta)
    A = y_hat - Y
    m = len(X)
    return 1/(2*m) * np.sum(np.matmul(A, A.T))

# Feature normalization 
def featureNormalize(X):
    X_norm = np.copy(X)
    mu = np.array([0.0 for i in range(X.shape[1])])
    sigma = np.array([0.0 for i in range(X.shape[1])])

    for i in range(X.shape[1]):
        feature = np.array(X[:,i])
        mu[i] = np.mean(feature)
        sigma[i] = np.std(feature)
        X_norm[:,i] = (feature - mu[i])/sigma[i]
    
    return X_norm, mu, sigma

# Gradient descent 
def gradientDescent(X, Y, theta, alpha, num_iter):
    J_history = [0 for i in range(num_iter)]
    m = len(Y)

    for i in range(num_iter):
        # Errors
        e = np.matmul(X, theta) - Y
        # Change in theta
        delta = 1/m * np.sum(np.multiply(e.reshape(-1, 1), X), axis=0)
        theta = theta - alpha * delta

        print(theta, computeCostJ(X, Y, theta))
    
    return theta, J_history


# Read data from CSV 
df = pd.read_csv("ex1data2.txt", header=None)

X = np.array(df[[0, 1]]).astype(float)
Y = np.array(df[2]).astype(float)

# Normalize data
X, mu, sigma = featureNormalize(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Run gradient descent 
theta = [0, 0, 0]
alpha = 0.0000001
num_iter = 400
theta, J_history = gradientDescent(X, Y, theta, alpha, num_iter)

