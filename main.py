from RandomForest import RandomForest

# Przykład użycia:
X = [[2.8, 1.0], [3.4, 2.1], [1.2, 0.5], [3.0, 1.5], [1.5, 0.7]]
y = [1, 1, 0, 1, 0]

# Tworzenie i trenowanie lasu losowego
forest = RandomForest(n_trees=5, max_depth=3)
forest.fit(X, y)

# Predykcja
predictions = forest.predict(X)
print(predictions)
