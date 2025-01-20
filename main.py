from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from RandomForest.RandomForest import RandomForest
from test import test_n_trees, test_depth, test_n_trees_with_comparison, test_depth_with_comparison, test_draw


file = "data/gym_members_exercise_tracking.csv"

test_draw(file, 3)

# Test 1: Liczba drzew przy stałej głębokości
n_trees_values = [1, 5, 10, 20, 30, 40, 50, 60]
max_depth = 10
#test_n_trees(file, n_trees_values, max_depth)
#test_n_trees_with_comparison(file, n_trees_values, max_depth)

# Test 2: Głębokość przy stałej liczbie drzew
n_trees = 12
depth_values = [2, 4, 6, 8, 10, 12, 14, 16]
#test_depth(file, depth_values, n_trees)
#test_depth_with_comparison(file, depth_values, n_trees)
