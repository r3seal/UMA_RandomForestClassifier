import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest.RandomForest import RandomForest

# Generowanie danych
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parametry testowe
n_trees_values = [1, 4, 7, 10, 13, 16, 19, 25, 30, 40, 50]
max_depth_values = [3, 5, 10, None]

# Przechowywanie wyników
results_sklearn = []
results_custom = []

# Testy dla liczby drzew i maksymalnej głębokości
for n_trees in n_trees_values:
    for max_depth in max_depth_values:
        # Testy dla sklearn
        start_time = time.time()
        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results_sklearn.append((n_trees, max_depth, accuracy, training_time))

        # Testy dla własnej implementacji
        start_time = time.time()
        custom_forest = RandomForest(n_trees=n_trees, max_depth=max_depth)
        custom_forest.fit(X_train.tolist(), y_train.tolist())
        training_time = time.time() - start_time
        predictions = custom_forest.predict(X_test.tolist())
        accuracy = accuracy_score(y_test.tolist(), predictions)
        results_custom.append((n_trees, max_depth, accuracy, training_time))

# Wykresy wyników
plt.figure(figsize=(12, 6))

# Wykres dokładności
plt.subplot(1, 2, 1)
for max_depth in max_depth_values:
    accuracies_sklearn = [r[2] for r in results_sklearn if r[1] == max_depth]
    accuracies_custom = [r[2] for r in results_custom if r[1] == max_depth]
    plt.plot(n_trees_values, accuracies_sklearn, label=f'sklearn max_depth={max_depth}', linestyle='--')
    plt.plot(n_trees_values, accuracies_custom, label=f'custom max_depth={max_depth}')
plt.xlabel('Liczba drzew')
plt.ylabel('Dokładność')
plt.title('Dokładność vs Liczba drzew')
plt.legend()

# Wykres czasu trenowania
plt.subplot(1, 2, 2)
for max_depth in max_depth_values:
    times_sklearn = [r[3] for r in results_sklearn if r[1] == max_depth]
    times_custom = [r[3] for r in results_custom if r[1] == max_depth]
    plt.plot(n_trees_values, times_sklearn, label=f'sklearn max_depth={max_depth}', linestyle='--')
    plt.plot(n_trees_values, times_custom, label=f'custom max_depth={max_depth}')
plt.xlabel('Liczba drzew')
plt.ylabel('Czas trenowania (s)')
plt.title('Czas trenowania vs Liczba drzew')
plt.legend()

plt.tight_layout()
plt.show()
