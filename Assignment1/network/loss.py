import numpy as np

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Computes the cross-entropy cost, with optional L2 regularization.
    """
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL+1e-8)) / m

    if lambd != 0 and parameters is not None:
        L2_cost = 0
        for l in range(1, len(parameters) // 2 + 1):
            L2_cost += np.sum(np.square(parameters[f"W{l}"]))
        cost += (lambd / (2 * m)) * L2_cost

    return cost
