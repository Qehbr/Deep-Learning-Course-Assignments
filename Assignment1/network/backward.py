import numpy as np
from .activations import relu_backward, softmax_backward


def linear_backward(dZ, cache, lambd=0.0):
    """
    Implements the linear part of the backward propagation process for a single layer.

    Parameters
    ----------
    dZ : ndarray
        The gradient of the cost with respect to the linear output of the current layer (layer l)
    cache : dict
        A dictionary containing A_prev, W, and b from the forward propagation of the current layer
    lambd : float, optional
        Regularization parameter (default is 0.0)

    Returns
    -------
    dA_prev : ndarray
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW : ndarray
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : ndarray
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[1]
    if lambd != 0:
        dW = (np.dot(dZ, A_prev.T) / m) + (lambd / m) * W
    else:
        dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd=0.0):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.

    Parameters
    ----------
    dA : ndarray
        Post-activation gradient of the current layer
    cache : dict
        Dictionary containing both the linear cache and the activation cache
    activation : str
        The activation function used in this layer ("relu" or "softmax")
    lambd : float, optional
        Regularization parameter (default is 0.0)

    Returns
    -------
    dA_prev : ndarray
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW : ndarray
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : ndarray
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache = {'A': cache['A'], 'W': cache['W'], 'b': cache['b']}
    activation_cache = {'Z': cache['Z']}

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    else:
        raise ValueError(f"Unsupported activation function: '{activation}'.")

    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)
    return dA_prev, dW, db


def l_model_backward(AL, Y, caches, lambd=0.0):
    """
    Implements the backward propagation process for the entire network.

    Parameters
    ----------
    AL : ndarray
        The probabilities vector, output of the forward propagation (L_model_forward)
    Y : ndarray
        The true labels vector (the "ground truth" - true classifications)
    caches : list of dict
        List of caches containing for each layer:
        a) the linear cache;
        b) the activation cache
    lambd : float, optional
        Regularization parameter (default is 0.0)

    Returns
    -------
    grads : dict
        Dictionary with the gradients
        grads["dA" + str(l)] -- Gradient of the cost with respect to the activation of layer l
        grads["dW" + str(l)] -- Gradient of the cost with respect to W of layer l
        grads["db" + str(l)] -- Gradient of the cost with respect to b of layer l
    """
    grads = {}
    L = len(caches)

    dAL = AL - Y
    current_cache = caches[L - 1]
    grads[f"dA{L}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(dAL, current_cache, "softmax", lambd)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l + 2}"], current_cache, "relu", lambd)
        grads[f"dA{l + 1}"] = dA_prev
        grads[f"dW{l + 1}"] = dW
        grads[f"db{l + 1}"] = db

    return grads
