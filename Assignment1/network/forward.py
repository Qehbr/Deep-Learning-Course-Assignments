import numpy as np

from .activations import relu, softmax
from .batchnorm import apply_batchnorm


def linear_forward(A, W, b):
    """
    Computes the linear part of a layer's forward propagation.
    """
    Z = np.dot(W, A) + b
    cache = {'A': A, 'W': W, 'b': b}
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, use_batchnorm=False):
    """
    Forward propagation for the LINEAR->ACTIVATION layer.
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if use_batchnorm:
        Z = apply_batchnorm(Z)
    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    else:
        raise ValueError(f"Unsupported activation function: '{activation}'. Expected 'relu' or 'softmax'.")

    cache = linear_cache | activation_cache
    return A, cache


def l_model_forward(X, parameters, use_batchnorm=False):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX model.
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], activation="relu",
                                             use_batchnorm=use_batchnorm)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters[f"W{L}"], parameters[f"b{L}"], activation="softmax"
    )
    caches.append(cache)

    return AL, caches
