from .Node import Node

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        classes = set(y)
        impurity = 1.0
        for c in classes:
            p = y.count(c) / len(y)
            impurity -= p ** 2
        return impurity

    def _split(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gini = None, None, float('inf')
        for feature in range(len(X[0])):
            values = set(row[feature] for row in X)
            for threshold in values:
                left_X, left_y, right_X, right_y = self._split(X, y, feature, threshold)
                gini = (len(left_y) / len(y)) * self._gini(left_y) + (len(right_y) / len(y)) * self._gini(right_y)
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Node(value=max(set(y), key=y.count))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=max(set(y), key=y.count))
        left_X, left_y, right_X, right_y = self._split(X, y, feature, threshold)
        return Node(
            feature=feature,
            value=threshold,
            left=self._build_tree(left_X, left_y, depth + 1),
            right=self._build_tree(right_X, right_y, depth + 1)
        )

    def predict_one(self, x, node):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.value:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]

    def draw(self):
        if self.root is None:
            print("Tree is empty.")
        else:
            print("\nDecision Tree Structure:\n")
            print(self._draw_node(self.root))

    def _draw_node(self, node, depth=0):
        indent = "    " * depth

        if node.feature is None:
            return f"{indent}Predict: {node.value}\n"

        result = f"{indent}Node\n"
        result += f"{indent}feature: {node.feature}\n"
        result += f"{indent}[X <= {node.value}]\n"
        result += f"{indent}├── If True:\n"
        result += self._draw_node(node.left, depth + 1)
        result += f"{indent}└── If False:\n"
        result += self._draw_node(node.right, depth + 1)

        return result