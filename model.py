import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def tanh_derivative(A):
    return 1 - np.power(A, 2)


def sigmoid_derivative(A, Y):
    return A - Y


activation_function_dict = {
    "tanh": tanh_derivative,
    "sigmoid": sigmoid_derivative,
}


# Define the size of the neural network with 1 hidden layer.
# Returns (input layer size, hidden layer size, output layer size)
def layer_sizes(X, Y, hidden_layer_node_count=4):
    n_x = X.shape[0]
    n_h = hidden_layer_node_count
    n_y = Y.shape[0]
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y, weight_multiple=0.01):
    W1 = np.random.randn(n_h, n_x) * weight_multiple
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * weight_multiple
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, model_parameters):
    W1 = model_parameters["W1"]
    b1 = model_parameters["b1"]
    W2 = model_parameters["W2"]
    b2 = model_parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    m_train = A2.shape[1]

    J = np.sum(- (Y * np.log(A2) + (1 - Y) * np.log(1 - A2))) / m_train

    return float(J)


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    # Outputs needed from forward-propagation needed to compute backward-propagation
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = sigmoid_derivative(A2, Y)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

    return gradients


def update_parameters(parameters, grads, learning_rate=1.2):
    parameters = copy.deepcopy(parameters)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, num_iterations=10000, print_cost=False):
    n_x, n_h, n_y = layer_sizes(X, Y, 4)
    model_params = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, model_params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(model_params, cache, X, Y)
        model_params = update_parameters(model_params, grads)

        # YOUR CODE ENDS HERE

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return model_params


def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    # classify to 0/1 using 0.5 as the threshold.
    predictions = (A2 > 0.5)

    return predictions


if __name__ == "__main__":
    X, Y = load_planar_dataset()
    m_train = X.shape[1]
    print("Number of training examples is " + str(m_train))
    # Build a model with a n_h-dimensional hidden layer
    parameters = nn_model(X, Y, num_iterations=10000, print_cost=True)

    # Print accuracy
    predictions = predict(X, parameters)
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
