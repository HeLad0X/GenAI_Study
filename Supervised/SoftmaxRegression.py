from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ---------- helpers ----------
def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def f(X, theta):
    return softmax(X @ theta)

def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1
    return Y

def log_likelihood(theta, X, Y):
    P = f(X, theta)
    return np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))

def loglike_gradient(theta, X, Y):
    P = f(X, theta)
    return X.T @ (Y - P) / len(X)

# ---------- data ----------
X, y = datasets.load_iris(return_X_y=True)
K = len(np.unique(y))

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.c_[X, np.ones(len(X))]   # bias

Y = one_hot(y, K)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=46
)

# ---------- training ----------
theta = np.zeros((X.shape[1], K))
theta_prev = np.ones_like(theta)

step_size = 0.1
threshold = 1e-6
iter = 0

while np.linalg.norm(theta - theta_prev) > threshold and iter < 100000:
    theta_prev = theta.copy()
    grad = loglike_gradient(theta, X_train, Y_train)
    theta += step_size * grad
    iter += 1

# ---------- evaluation ----------
P_test = f(X_test, theta)
print("Multiclass log loss:", log_loss(np.argmax(Y_test, axis=1), P_test))
