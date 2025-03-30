import numpy as np

def apply_batchnorm(A):
    """
    Normalizes activation values of a given layer using batch normalization.
    """
    mu = np.mean(A, axis=1, keepdims=True)
    sigma = np.std(A, axis=1, keepdims=True)
    A_norm = (A - mu) / (sigma+1e-8)
    return A_norm
