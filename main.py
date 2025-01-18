from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from RandomForest.RandomForest import RandomForest
from test import test

X, y = make_classification(
    n_samples=20,  # Więcej próbek dla bardziej złożonego drzewa
    n_features=3,  # Liczba cech
    n_classes=2,   # Liczba klas (binarny problem)
    n_informative=3,
    n_redundant=0,
    random_state=42
)

# Tworzenie i trenowanie lasu losowego
forest = RandomForest(n_trees=5, max_depth=4)  # Ustawienie większej głębokości drzewa
forest.fit(X, y)

# Wypisanie struktury pierwszego drzewa w lesie
print("Pierwsze drzewo w lesie losowym:")
forest.trees[0].draw()


# # Predykcja
# predictions = forest.predict(X)
# print(predictions)
#
# # Generowanie danych
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# test(X_train, X_test, y_train, y_test)
