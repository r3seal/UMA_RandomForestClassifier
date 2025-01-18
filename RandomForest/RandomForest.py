import random

from .DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        errors = [False] * n_samples
        for _ in range(self.n_trees):
            # Podział danych na poprawne i niepoprawne predykcje
            correct = [i for i in range(n_samples) if not errors[i]]
            incorrect = [i for i in range(n_samples) if errors[i]]
            indices = random.choices(correct + incorrect * 2, k=n_samples)
            sample_X = [X[i] for i in indices]
            sample_y = [y[i] for i in indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

            # Aktualizacja błędów
            predictions = tree.predict(X)
            errors = [predictions[i] != y[i] for i in range(n_samples)]

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        final_predictions = []
        for i in range(len(X)):
            votes = [pred[i] for pred in predictions]
            final_predictions.append(max(set(votes), key=votes.count))
        return final_predictions