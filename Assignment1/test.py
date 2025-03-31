from data.mnist_loader import load_mnist
from train.model import l_layer_model
from train.predict import predict
from train.utils import plot_costs

def run(learning_rate, batch_size, use_batchnorm, lambd):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()
    layers_dims = [784, 20, 7, 5, 10]

    parameters, costs = l_layer_model(
        X_train, Y_train,
        layers_dims=layers_dims,
        learning_rate=learning_rate,
        num_iterations=3000,
        batch_size=batch_size,
        use_batchnorm=use_batchnorm,
        lambd=lambd
    )

    print("Train accuracy:", predict(X_train, Y_train, parameters))
    print("Validation accuracy:", predict(X_val, Y_val, parameters))
    print("Test accuracy:", predict(X_test, Y_test, parameters))
    plot_costs(costs)

run(0.001, 16, use_batchnorm=False, lambd=0.0)