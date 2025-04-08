import numpy as np


def apply_batchnorm(A):
    """
    Performs batch normalization on the activation values of a given layer.

    Parameters
    ----------
    A : ndarray
        The activation values of a given layer

    Returns
    -------
    NA : ndarray
        The normalized activation values, based on the batch normalization formula
    """

    mu = np.mean(A, axis=1, keepdims=True)
    variance = np.std(A, axis=1, keepdims=True)
    NA = (A - mu) / np.sqrt(variance + 1e-8)
    return NA
