import numpy as np

def batchnorm_forward(Z, epsilon=1e-8):
    """
    Performs the forward pass for batch normalization (without gamma/beta).

    Args:
        Z (np.array): Input data (output of the linear layer), shape (num_features, num_samples)
        epsilon (float): Small constant for numerical stability.

    Returns:
        Z_norm (np.array): Normalized data.
        cache (tuple): Values needed for backward pass (Z, mu, variance, Z_centered, inv_std_dev, epsilon).
    """
    # Calculate mean and variance over the batch dimension (axis=1)
    mu = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)

    # Normalize
    Z_centered = Z - mu
    inv_std_dev = 1. / np.sqrt(variance + epsilon)
    Z_norm = Z_centered * inv_std_dev

    # Cache values needed for backward pass
    cache = (Z, mu, variance, Z_centered, inv_std_dev, epsilon)

    return Z_norm, cache

def batchnorm_backward(dZ_norm, cache):
    """
    Performs the backward pass for batch normalization (without gamma/beta).

    Args:
        dZ_norm (np.array): Gradient of the cost w.r.t. the output of batch norm.
        cache (tuple): Values from the forward pass.

    Returns:
        dZ (np.array): Gradient of the cost w.r.t. the input of batch norm (output of linear layer).
    """
    Z, mu, variance, Z_centered, inv_std_dev, epsilon = cache
    m = Z.shape[1] # Number of samples in the batch

    # Intermediate gradients (following standard BN backward derivation, simplified for no gamma/beta)
    dZ_centered = dZ_norm * inv_std_dev
    dvariance = np.sum(dZ_norm * Z_centered * (-0.5) * (inv_std_dev**3), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * (-inv_std_dev), axis=1, keepdims=True) + dvariance * np.mean(-2.0 * Z_centered, axis=1, keepdims=True)

    # Final gradient w.r.t. Z
    dZ = dZ_centered + (dvariance * 2.0 * Z_centered / m) + (dmu / m)

    return dZ
