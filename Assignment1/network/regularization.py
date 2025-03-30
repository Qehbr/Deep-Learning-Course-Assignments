import numpy as np

def compute_l2_cost(parameters, lambd):
    """
    Computes the L2 regularization cost term.
    """
    L2_cost = 0
    for l in range(1, len(parameters) // 2 + 1):
        L2_cost += np.sum(np.square(parameters[f"W{l}"]))
    return L2_cost * lambd / 2

def compute_l2_gradients(parameters, lambd, m):
    """
    Computes L2 gradients to be added to dW during backprop.
    """
    grads = {}
    for l in range(1, len(parameters) // 2 + 1):
        grads[f"dW{l}"] = (lambd / m) * parameters[f"W{l}"]
    return grads
