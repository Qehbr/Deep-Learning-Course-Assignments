def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters
