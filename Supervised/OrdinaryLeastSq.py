import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True, scaled=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

theta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
theta_best_df = pd.DataFrame(data=theta[np.newaxis, :], columns=X_train.columns)

print(theta_best_df)

y_test_pred = X_test.dot(theta)

print(y_test.head())
print(y_test_pred.head())

linear = LinearRegression()
linear.fit(X_train, y_train)

y_test_pred = linear.predict(X_test)

print(y_test.head())
print(y_test_pred[:5])