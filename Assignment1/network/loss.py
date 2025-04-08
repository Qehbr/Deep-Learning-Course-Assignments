import numpy as np


def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Computes the cross-entropy cost, with optional L2 regularization.

    Parameters
    ----------
    AL : ndarray
        Probability vector corresponding to your label predictions,
        of shape (number of classes, number of examples)
    Y : ndarray
        The labels vector (i.e. the ground truth), same shape as AL
    parameters : dict, optional
        Dictionary containing weights and biases of each layer. Required only if L2 regularization is used
    lambd : float, optional
        L2 regularization hyperparameter. If set to 0.0 (default), no regularization is applied

    Returns
    -------
    cost : float
        The cross-entropy cost (with L2 regularization if lambd > 0)

    Notes
    -----
    The categorical cross-entropy loss is computed as:
        cost = -1/m * sum(sum(Y * log(AL)))
    where `m` is the number of examples.

    If `lambd` > 0, an additional L2 regularization term is added:
        L2_cost = (lambd / 2m) * sum(||W[l]||^2) for all layers l
    """
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL + 1e-8)) / m

    if lambd != 0 and parameters is not None:
        L2_cost = 0
        for l in range(1, len(parameters) // 2 + 1):
            L2_cost += np.sum(np.square(parameters[f"W{l}"]))
        cost += (lambd / (2 * m)) * L2_cost

    return cost
