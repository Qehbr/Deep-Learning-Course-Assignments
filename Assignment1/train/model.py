import time
import numpy as np
from network.parameters import initialize_parameters
from network.forward import l_model_forward
from network.backward import l_model_backward
from network.loss import compute_cost
from network.optimization import update_parameters
from train.predict import predict
from .utils import create_minibatches


def l_layer_model(
        X, Y,
        layers_dims,
        learning_rate,
        num_iterations,
        batch_size,
        use_batchnorm,
        lambd,
        tolerance,
        patience,
        random_seed=1
):
    """
    Implements a L-layer neural network. All layers but the last use ReLU activation,
    and the final layer uses softmax. The output layer size should match the number of labels.

    Parameters
    ----------
    X : ndarray
        Input data, of shape (height * width, number_of_examples)
        (grayscale input: height * width)
    Y : ndarray
        Ground truth labels, of shape (num_of_classes, number_of_examples)
    layers_dims : list of int
        Dimensions of each layer in the network, including input and output
    learning_rate : float
        The learning rate for gradient descent
    num_iterations : int
        Total number of training iterations
    batch_size : int
        Number of training examples in a single minibatch
    use_batchnorm : bool
        If True, applies batch normalization after activations
    lambd : float
        L2 regularization parameter
    tolerance : float
        Minimum required improvement in validation cost to reset early stopping counter
    patience : int
        Number of validation checks to wait before early stopping if no improvement
    random_seed : int, optional
        Seed for random operations to ensure reproducibility (default is 1)

    Returns
    -------
    parameters : dict
        Learned parameters of the network
    costs : list of tuple
        Cost values logged every 100 iterations as (iteration, avg_train_cost, val_cost)
    """

    np.random.seed(random_seed)
    parameters = initialize_parameters(layers_dims)

    # split training and validation set
    m = X.shape[1]
    val_size = int(m * 0.2)
    indices = np.random.permutation(m)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train, Y_train = X[:, train_idx], Y[:, train_idx]
    X_val, Y_val = X[:, val_idx], Y[:, val_idx]

    best_val_cost = float('inf')
    patience_counter = 0
    start_time = time.time()

    costs = []  # stores tuples (iteration, avg_train_cost, val_cost)

    iteration = 0
    epoch = 0
    training_cost_sum = 0.0  # sums up the cost for the last 100 minibatches
    training_count = 0  # how many minibatches have we counted toward training_cost_sum

    while iteration < num_iterations:
        minibatches = create_minibatches(X_train, Y_train, batch_size)

        for minibatch_X, minibatch_Y in minibatches:
            if iteration >= num_iterations:
                break

            AL_train, caches = l_model_forward(minibatch_X, parameters, use_batchnorm)
            cost = compute_cost(AL_train, minibatch_Y, parameters, lambd)
            grads = l_model_backward(AL_train, minibatch_Y, caches, lambd)
            parameters = update_parameters(parameters, grads, learning_rate)

            training_cost_sum += cost
            training_count += 1
            iteration += 1

            if iteration % 100 == 0:
                avg_train_cost = training_cost_sum / training_count

                AL_val, _ = l_model_forward(X_val, parameters, use_batchnorm)
                val_cost = compute_cost(AL_val, Y_val, parameters, lambd)

                costs.append((iteration, avg_train_cost, val_cost))
                print(f"Iteration {iteration}: "
                      f"Train cost = {avg_train_cost:.4f}, "
                      f"Val cost = {val_cost:.4f}")

                training_cost_sum = 0.0
                training_count = 0

                # early stopping check
                if val_cost + tolerance < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {iteration}, "
                              f"no significant improvement in validation cost.")
                        print(f"Total epochs: {epoch} (approx), "
                              f"Total iterations: {iteration}")
                        print(f"Total training time: {time.time() - start_time:.2f} seconds")
                        print("Train accuracy:", predict(X_train, Y_train, parameters, use_batchnorm))
                        print("Validation accuracy:", predict(X_val, Y_val, parameters, use_batchnorm))
                        return parameters, costs

        epoch += 1

    print(f"Finished training at iteration {iteration}, epoch {epoch}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print("Train accuracy:", predict(X_train, Y_train, parameters, use_batchnorm))
    print("Validation accuracy:", predict(X_val, Y_val, parameters, use_batchnorm))

    return parameters, costs
