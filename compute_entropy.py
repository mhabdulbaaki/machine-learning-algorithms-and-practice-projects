import numpy as np


def compute_entropy(y):
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """

    entropy = 0.

    if len(y) == 0:
        return 0

    p_1 = len(y[y == 1]) / len(y)
    p_0 = 1 - p_1

    if p_1 != 0 and p_1 != 1:
        h_1 = p_1 * np.log2(p_1)
        h_0 = p_0 * np.log2(p_0)

        entropy = -h_0 - h_1

    else:
        entropy = 0

    return entropy
