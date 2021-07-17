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

    for k in range(num_iter):
        theta -= alpha * 1/m * X.T @ (X @ theta - Y)
        J_history[k] = computeCostJ(X, Y, theta)
    
    return theta, J_history


# Read data from CSV 
df = pd.read_csv("ex1data2.txt", header=None)

X = np.array(df[[0, 1]], dtype="double")
Y = np.array(df[2], dtype="double")

# Normalize data
X, mu, sigma = featureNormalize(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Run gradient descent 
theta = np.array([0, 0, 0], dtype="double")
alpha = 0.1
num_iter = 200
theta, J_history = gradientDescent(X, Y, theta, alpha, num_iter)

# Test 
x = np.array([1, 1650, 3])
print("Predict the price of a house having S=%d and %d bedrooms: " % (x[1], x[2]), end="")
# Normalize input 
sigma = np.hstack((1, sigma))
mu = np.hstack((0, mu))
x = (x - mu) / sigma
# Predict 
price = x @ theta
print("$%f" % price)

# Draw iteration
plt.plot(np.linspace(1, num_iter, num_iter), J_history)
plt.xlabel("Num. iters")
plt.ylabel("Cost function")
plt.show()