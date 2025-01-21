from test import test_n_trees, test_depth, test_n_trees_with_comparison, test_depth_with_comparison, test_draw


file = "data/gym_members_exercise_tracking.csv"

test_draw(file, 4)

# Test 1: Liczba drzew przy stałej głębokości
n_trees_values = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
max_depth = 6
test_n_trees(file, n_trees_values, max_depth)
test_n_trees_with_comparison(file, n_trees_values, max_depth)

# Test 2: Głębokość przy stałej liczbie drzew
# Test 2: Głębokość przy stałej liczbie drzew
n_trees = 6
depth_values = [1, 2, 3, 4, 5, 6, 7, 8]
test_depth(file, depth_values, n_trees)
test_depth_with_comparison(file, depth_values, n_trees)
