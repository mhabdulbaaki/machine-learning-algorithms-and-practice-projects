from compute_information_gain import compute_information_gain


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1
    max_info_gain = 0

    for i in range(num_features):
        info_gain_i = compute_information_gain(X, y, node_indices, feature=i)

        if info_gain_i > max_info_gain:
            max_info_gain = info_gain_i
            best_feature = i

    return best_feature
