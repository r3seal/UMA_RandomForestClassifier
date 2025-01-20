import time
import matplotlib.pyplot as plt
import csv
from RandomForest.RandomForest import RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from math import sqrt


# Wczytanie danych z pliku CSV
def load_data(file_path):
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            data.append(row)
    return data


# Przygotowanie danych w formacie X i y
def prepare_data(data):
    X, y = [], []
    for row in data:
        # Wybrane kolumny jako cechy (X) oraz Workout_Type jako label (y)
        X.append([
            float(row["Age"]),
            1 if row["Gender"] == "Male" else 0,
            float(row["Weight (kg)"]),
            float(row["Height (m)"]),
            float(row["Max_BPM"]),
            float(row["Avg_BPM"]),
            float(row["Resting_BPM"]),
            float(row["Session_Duration (hours)"]),
            float(row["Calories_Burned"]),
            float(row["Fat_Percentage"]),
            float(row["Water_Intake (liters)"]),
            float(row["Workout_Frequency (days/week)"]),
            float(row["Experience_Level"]),
            float(row["BMI"]),
        ])
        y.append(row["Workout_Type"])
    return X, y


# Funkcja do przeprowadzenia testów
def test_random_forest(X, y, values_to_test, const_param, test_type="n_trees"):
    accuracies = []
    times = []

    for param in values_to_test:
        if test_type == "n_trees":
            n_trees = param
            depth = const_param
        elif test_type == "depth":
            n_trees = const_param
            depth = param

        # Trening modelu i pomiar czasu
        start_time = time.time()
        model = RandomForest(n_trees=n_trees, max_depth=depth)
        model.fit(X, y)
        train_time = time.time() - start_time

        # Predykcja i obliczenie skuteczności
        predictions = model.predict(X)
        accuracy = sum(1 for i in range(len(y)) if predictions[i] == y[i]) / len(y)

        accuracies.append(accuracy)
        times.append(train_time)

    return accuracies, times


def test_random_forest_comparison(X, y, values_to_test, const_param, test_type="n_trees"):
    accuracies_custom = []
    accuracies_sklearn = []

    for param in values_to_test:
        if test_type == "n_trees":
            n_trees = param
            depth = const_param
        elif test_type == "depth":
            n_trees = const_param
            depth = param

        # nasza implementacja
        custom_model = RandomForest(n_trees=n_trees, max_depth=depth)
        custom_model.fit(X, y)
        custom_predictions = custom_model.predict(X)
        custom_accuracy = sum(1 for i in range(len(y)) if custom_predictions[i] == y[i]) / len(y)
        accuracies_custom.append(custom_accuracy)

        # gotowa implementacja sklearn
        sklearn_model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, random_state=42, max_samples=int(sqrt(len(X))))
        sklearn_model.fit(X, y)
        sklearn_predictions = sklearn_model.predict(X)
        sklearn_accuracy = accuracy_score(y, sklearn_predictions)
        accuracies_sklearn.append(sklearn_accuracy)

    return accuracies_custom, accuracies_sklearn


# Rysowanie wykresów
def plot_results(x_values, accuracies, times, x_label, test_type):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy", color=color)
    ax1.plot(x_values, accuracies, marker="o", color=color, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Training Time (s)", color=color)
    ax2.plot(x_values, times, marker="x", linestyle="--", color=color, label="Training Time")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.title(f"Random Forest {test_type.capitalize()} Test")
    plt.show()


def plot_comparison(values, accuracies_custom, accuracies_sklearn, xlabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(values, accuracies_custom, label="Custom RandomForest", marker='o', color='blue')
    plt.plot(values, accuracies_sklearn, label="Sklearn RandomForest", marker='o', color='orange')
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Główna funkcja testowa
def test_draw(file, max_depth):
    data = load_data(file)
    X, y = prepare_data(data)

    print("Przykładowe drzewo")
    model = RandomForest(n_trees=1, max_depth=max_depth)
    model.fit(X, y)
    model.trees[0].draw()

def test_n_trees(file, n_trees_values, max_depth):
    data = load_data(file)
    X, y = prepare_data(data)

    print("Test 1: Liczba drzew przy stałej głębokości")
    accuracies, times = test_random_forest(X, y, n_trees_values, max_depth, test_type="n_trees")
    plot_results(n_trees_values, accuracies, times, "Number of Trees", "n_trees")


def test_depth(file, depth_values, n_trees):
    data = load_data(file)
    X, y = prepare_data(data)

    print("Test 2: Głębokość przy stałej liczbie drzew")
    accuracies, times = test_random_forest(X, y, depth_values, n_trees, test_type="depth")
    plot_results(depth_values, accuracies, times, "Max Depth", "depth")


def test_n_trees_with_comparison(file, n_trees_values, max_depth):
    data = load_data(file)
    X, y = prepare_data(data)

    print("Test 1: Liczba drzew przy stałej głębokości")
    accuracies_custom, accuracies_sklearn = test_random_forest_comparison(
        X, y, n_trees_values, max_depth, test_type="n_trees"
    )
    plot_comparison(
        n_trees_values, accuracies_custom, accuracies_sklearn, "Number of Trees",
        "Comparison: Number of Trees vs Accuracy"
    )


def test_depth_with_comparison(file, depth_values, n_trees):
    data = load_data(file)
    X, y = prepare_data(data)

    print("Test 2: Głębokość przy stałej liczbie drzew")
    accuracies_custom, accuracies_sklearn = test_random_forest_comparison(
        X, y, depth_values, n_trees, test_type="depth"
    )
    plot_comparison(
        depth_values, accuracies_custom, accuracies_sklearn, "Max Depth",
        "Comparison: Max Depth vs Accuracy"
    )