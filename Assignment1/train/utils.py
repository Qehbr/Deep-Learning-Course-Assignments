import numpy as np
import matplotlib.pyplot as plt

def create_minibatches(X, Y, batch_size):
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    minibatches = []
    num_complete_minibatches = m // batch_size
    for k in range(num_complete_minibatches):
        X_batch = X_shuffled[:, k*batch_size:(k+1)*batch_size]
        Y_batch = Y_shuffled[:, k*batch_size:(k+1)*batch_size]
        minibatches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_minibatches*batch_size:]
        Y_batch = Y_shuffled[:, num_complete_minibatches*batch_size:]
        minibatches.append((X_batch, Y_batch))

    return minibatches

def plot_costs(costs):

    steps, cost_values = zip(*costs)
    plt.plot(steps, cost_values)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost over iterations")
    plt.grid(True)
    plt.show()