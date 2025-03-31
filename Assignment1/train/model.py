import numpy as np
from network.parameters import initialize_parameters
from network.forward import l_model_forward
from network.backward import l_model_backward
from network.loss import compute_cost
from network.optimization import update_parameters
from train.predict import predict
from .utils import create_minibatches


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations,
                  batch_size, use_batchnorm, lambd, tolerance, random_seed=1):
    np.random.seed(random_seed)
    costs = []  # List of tuples: (iteration, train_cost, val_cost)
    parameters = initialize_parameters(layers_dims)

    # Split training and validation set
    m = X.shape[1]
    val_size = int(m * 0.2)
    indices = np.random.permutation(m)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train, Y_train = X[:, train_indices], Y[:, train_indices]
    X_val, Y_val = X[:, val_indices], Y[:, val_indices]

    best_val_cost = float('inf')
    patience_counter = 0

    for i in range(num_iterations):
        minibatches = create_minibatches(X_train, Y_train, batch_size)
        epoch_cost = 0

        for minibatch_X, minibatch_Y in minibatches:
            AL_train, caches = l_model_forward(minibatch_X, parameters, use_batchnorm)
            cost = compute_cost(AL_train, minibatch_Y, parameters, lambd)
            grads = l_model_backward(AL_train, minibatch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            epoch_cost += cost / len(minibatches)

        # Compute validation cost
        AL_val, _ = l_model_forward(X_val, parameters, use_batchnorm)
        val_cost = compute_cost(AL_val, Y_val, parameters, lambd)

        costs.append((i, epoch_cost, val_cost))
        print(f"Iteration {i}: Train cost = {epoch_cost:.4f}, Val cost = {val_cost:.4f}")

        # Early stopping
        if val_cost + tolerance < best_val_cost:
            best_val_cost = val_cost
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 100:
                print(f"Early stopping at iteration {i}, no significant improvement in validation cost.")
                break

    print("Train accuracy:", predict(X_train, Y_train, parameters))
    print("Validation accuracy:", predict(X_val, Y_val, parameters))


    return parameters, costs
