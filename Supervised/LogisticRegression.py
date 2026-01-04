from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def f(X, theta):
    return sigmoid(X.dot(theta))

def log_likelihood(theta, X, y):
    return (y * np.log(f(X, theta) + 1e-6) + (1-y) * np.log(1 - f(X, theta) + 1e-6)).mean()

def loglike_gradient(theta, X, y):
    preds = f(X, theta)
    return X.T @ (y - preds) / len(y)



opt_grads = []

iter = 0

X, y = datasets.load_iris(return_X_y= True, as_frame=True)
X['one'] = 1
y = (y == 2).astype(int)

scaler = StandardScaler()
scaled_x = scaler.fit_transform(X)

theta = np.zeros(X.shape[1])
theta_prev = np.ones(X.shape[1])

opt_pts = [theta]

X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=46)
threshold = 5e-7
step_size = 1e-1
while np.linalg.norm(theta - theta_prev) > threshold and iter < 1000000:
    if iter % 1000 == 0:
        print(f"Iteration {iter}. Log-likelihood: {log_likelihood(theta, X_train, y_train)}")

    theta_prev = theta.copy()
    gradient = loglike_gradient(theta, X_train, y_train)
    theta = theta_prev + step_size * gradient
    opt_grads += [gradient]
    opt_pts += [theta]
    iter += 1

y_pred = f(X_test, theta)
print("Log loss of prediction: ",log_loss(y_test, y_pred))