import numpy as np
from build_tree_recursive import build_tree_recursive


X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == '__main__':
    build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
