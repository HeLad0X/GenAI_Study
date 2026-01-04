import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


class Node:
    def __init__(self, is_leaf=False, prediction=None, best_feature=None,
                 threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.best_feature = best_feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, max_depth=2):
        self.head = None
        self.max_depth = max_depth

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if len(y) == 0:
            return Node(is_leaf=True, prediction=None)

        temp_df = X.copy()
        temp_df["target"] = y.to_numpy()

        if y.nunique() == 1:
            return Node(is_leaf=True, prediction=y.iloc[0])

        if depth == self.max_depth:
            return Node(is_leaf=True, prediction=y.mode().iloc[0])

        best_feature, best_threshold, best_gini = self.feature_average(temp_df)

        if best_feature is None or best_threshold is None:
            return Node(is_leaf=True, prediction=y.mode().iloc[0])

        left_df = temp_df[temp_df[best_feature] <= best_threshold]
        right_df = temp_df[temp_df[best_feature] > best_threshold]

        if left_df.empty or right_df.empty:
            return Node(is_leaf=True, prediction=y.mode().iloc[0])

        left_x = left_df.drop(columns=["target"])
        right_x = right_df.drop(columns=["target"])

        left_y = left_df["target"]
        right_y = right_df["target"]

        left_node = self.build_tree(left_x, left_y, depth=depth + 1)
        right_node = self.build_tree(right_x, right_y, depth=depth + 1)

        return Node(
            is_leaf=False,
            prediction=None,
            best_feature=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )

    def fit(self, X, y):
        self.head = self.build_tree(X, pd.Series(y), 0)

    def get_gini(self, target_arr):
        class_dict = {val: target_arr.count(val) for val in set(target_arr)}
        total = len(target_arr)
        if total == 0:
            return 0.0

        gini = 1.0
        for _, count in class_dict.items():
            gini -= (count / total) ** 2
        return gini

    def feature_average(self, df: pd.DataFrame):
        if len(df) <= 1:
            return None, None, 1

        best_feature, best_threshold, best_gini = None, None, 1

        for feature in df.drop(columns=["target"]).columns:
            sorted_df = df.sort_values(by=feature)
            sorted_y = sorted_df["target"].tolist()
            sorted_feature = sorted_df[feature].tolist()

            n = len(sorted_feature)
            for i in range(n - 1):
                if sorted_feature[i] == sorted_feature[i + 1]:
                    continue

                left_part = sorted_y[: i + 1]
                right_part = sorted_y[i + 1 :]

                if len(left_part) == 0 or len(right_part) == 0:
                    continue

                threshold = (sorted_feature[i] + sorted_feature[i + 1]) / 2
                gini_left = self.get_gini(left_part)
                gini_right = self.get_gini(right_part)

                weighted = gini_left * (i + 1) / n + gini_right * (n - i - 1) / n

                if weighted < best_gini:
                    best_gini = weighted
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold, best_gini

    def predict(self, x: pd.DataFrame):
        node = self.head

        if node is None:
            return None

        while not node.is_leaf:
            if node.best_feature is None or node.threshold is None:
                return node.prediction

            val = x[node.best_feature].iloc[0]
            if val <= node.threshold:
                if node.left is None:
                    return node.prediction
                node = node.left
            else:
                if node.right is None:
                    return node.prediction
                node = node.right

        return node.prediction

    def predict_many(self, X: pd.DataFrame):
        return [self.predict(X.iloc[[i]]) for i in range(len(X))]


if __name__ == "__main__":
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)

    scaler = StandardScaler()
    X = X.copy()
    X.fillna(X.mean(numeric_only=True), inplace=True)
    columns = X.columns
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(data=X_scaled, columns=columns)

    y = pd.Series(y).copy()
    y.fillna(y.mode().iloc[0], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=20, random_state=46
    )

    depths = [1, 2, 3, 4, 5, 6]
    best_depth = None
    best_accuracy = 0
    for depth in depths:
        tree = DecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)
        prediction = tree.predict_many(X_test)

        accuracy = accuracy_score(y_test, prediction)
        print(f"Depth: {depth}. Accuracy:", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth


    print(f'Best accuracy: {best_accuracy} reached at depth: {best_depth}')
