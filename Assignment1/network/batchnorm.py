import numpy as np

def apply_batchnorm(A):
    """
    Normalizes activation values of a given layer using batch normalization.
    """
    mu = np.mean(A, axis=1, keepdims=True)
    variance = np.std(A, axis=1, keepdims=True)
    NA = (A - mu) / np.sqrt(variance+1e-8)
    return NA
