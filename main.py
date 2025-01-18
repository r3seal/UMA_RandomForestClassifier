from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from test import test

# Przykład użycia:
X = [[2.8, 1.0], [3.4, 2.1], [1.2, 0.5], [3.0, 1.5], [1.5, 0.7]]
y = [1, 1, 0, 1, 0]

# Tworzenie i trenowanie lasu losowego
forest = RandomForest(n_trees=5, max_depth=3)
forest.fit(X, y)

# Predykcja
predictions = forest.predict(X)
print(predictions)

# Generowanie danych
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
test(X_train, X_test, y_train, y_test)
