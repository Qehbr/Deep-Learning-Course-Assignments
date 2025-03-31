import numpy as np

from .activations import relu, softmax
from .batchnorm import batchnorm_forward


def linear_forward(A, W, b):
    """
    Computes the linear part of a layer's forward propagation.
    """
    Z = np.dot(W, A) + b
    cache = {'A': A, 'W': W, 'b': b}
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, use_batchnorm=False):
    """
    Forward propagation for the LINEAR->[BATCHNORM]->ACTIVATION layer.
    """
    # 1. Linear step
    Z, linear_cache = linear_forward(A_prev, W, b)

    bn_cache = None # Initialize bn_cache
    # 2. Batch Normalization (Optional)
    if use_batchnorm:
        # Apply BN *before* activation
        Z, bn_cache = batchnorm_forward(Z) # Use the new function

    # 3. Activation step
    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        # Softmax is usually applied at the output layer without BN right before it
        # but if needed for an intermediate layer, BN would typically come before.
        # For the final layer, BN is less common. Assuming no BN before final softmax here.
        if use_batchnorm and bn_cache is not None:
             print("Warning: Applying BatchNorm before Softmax might not be standard.")
        A, activation_cache = softmax(Z)
    else:
        raise ValueError(f"Unsupported activation function: '{activation}'. Expected 'relu' or 'softmax'.")

    # Combine all caches for this layer
    # Store them distinctly for clarity in the backward pass
    cache = {'linear_cache': linear_cache, 'bn_cache': bn_cache, 'activation_cache': activation_cache}
    return A, cache


def l_model_forward(X, parameters, use_batchnorm=False):
    """
    Forward propagation for the [LINEAR->[BN]->RELU]*(L-1)->LINEAR->SOFTMAX model.
    """
    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden Layers with ReLU
    for l in range(1, L):
        A_prev = A
        # Pass use_batchnorm flag to the function for hidden layers
        A, cache = linear_activation_forward(
            A_prev, parameters[f"W{l}"], parameters[f"b{l}"],
            activation="relu",
            use_batchnorm=use_batchnorm # <-- Pass flag here
        )
        caches.append(cache)

    # Output Layer with Softmax
    # Typically, BN is NOT used right before the final output softmax layer
    AL, cache = linear_activation_forward(
        A, parameters[f"W{L}"], parameters[f"b{L}"],
        activation="softmax",
        use_batchnorm=False # <-- Explicitly False for final layer (common practice)
    )
    caches.append(cache)

    return AL, caches