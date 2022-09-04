def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    left_indices = []
    right_indices = []

    for i in node_indices:
        v = X[i][feature]
        if v == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices
