               
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn import svm


def f(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X @ theta


def svm_objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 0.1) -> float:
    
    margins = y * f(X, theta)
    loss = np.maximum(1.0 - margins, 0.0).mean()
    reg = 0.5 * C * np.sum(theta[:-1] ** 2)
    return loss + reg


def svm_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 0.1) -> np.ndarray:
    
    margins = y * f(X, theta)
    active = margins < 1.0  

    grad = np.zeros_like(theta)
    if np.any(active):
        grad += -(X[active].T @ y[active]) / X.shape[0]

    grad[:-1] += C * theta[:-1]
    return grad


def main():
    X_df, y_ser = datasets.load_iris(return_X_y=True, as_frame=True)

    # Keep only two classes
    mask = (y_ser == 2) | (y_ser == 1)
    X_df = X_df[mask]
    y_ser = y_ser[mask]

    # Convert labels to {-1, +1}
    y = y_ser.to_numpy().astype(int)
    y = np.where(y == 0, -1, 1)

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.to_numpy())
    X = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])  # last col = bias

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=46, shuffle=True
    )

    C = 0.1
    threshold = 5e-5
    step_size = 1e-2
    max_iters = 200000

    theta = np.ones(X_train.shape[1])
    theta_prev = np.zeros_like(theta)

    iters = 0
    while np.linalg.norm(theta - theta_prev) > threshold and iters < max_iters:
        if iters % 1000 == 0:
            print(f"Iteration {iters}. J: {svm_objective(theta, X_train, y_train, C=C):.6f}")

        theta_prev = theta.copy()
        theta -= step_size * svm_gradient(theta, X_train, y_train, C=C)
        iters += 1


    
    scores = f(X_test, theta)
    pred = np.sign(scores)
    pred[pred == 0] = 1  # in the rare case exactly 0, pick +1

    print("\nConfusion matrix (labels [-1, +1] in that order):")
    print(confusion_matrix(y_test, pred, labels=[-1, 1]))

    print("\nClassification report:")
    print(classification_report(y_test, pred, labels=[-1, 1], target_names=["class 0 (setosa)", "class 1 (versicolor)"]))


if __name__ == "__main__":
    main()
