import numpy as np
import json
import logging as log
import copy
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation

def tanh_derivative(A):
    return 1 - np.power(A, 2)

def sigmoid_derivative(A, Y):
    return A - Y

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def ReLU(x):
    return np.maximum(x, [0])

def ReLU_derivative(Z):
    derivative = np.zeros_like(Z)
    derivative[Z > 0] = 1.0
    return derivative

def normalize_rows(x):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x


def load_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data//test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def flatten_input_images(train_set_x_orig, test_set_x_orig):
    assert (train_set_x_orig.shape == (209, 64, 64, 3))

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]

    train_set_x = train_set_x_orig.reshape(m_train, -1).T
    test_set_x = test_set_x_orig.reshape(m_test, -1).T

    return train_set_x, test_set_x


def normalize_image_data(set_x):
    set_x = set_x / 255
    return set_x


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
    A1 = ReLU(Z1)
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
    Z1 = cache["Z1"]

    dZ2 = sigmoid_derivative(A2, Y)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims = True) / m
    dZ1 = np.dot(W2.T, dZ2) * ReLU_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    gradients = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return gradients


def update_parameters(parameters, grads, learning_rate=0.5):
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

def nn_model(X, Y, num_iterations = 10_000, learning_rate = 0.5, print_cost=False):
    n_x, n_h, n_y = layer_sizes(X, Y, 1000)
    print("Layer 0 size: " + str(n_x))
    print("Layer 1 size: " + str(n_h))
    print("Layer 2 size: " + str(n_y))
    model_params = initialize_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, model_params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(model_params, cache, X, Y)
        model_params = update_parameters(model_params, grads, learning_rate)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    return model_params, costs


def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    # classify to 0/1 using 0.5 as the threshold.
    predictions = np.squeeze(A2 > 0.5)
    predictions = [float(value) for value in predictions]

    return predictions

def plot_cost_function(costs):
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    ln, = plt.plot([], [], 'r-', animated=True)

    def init():
        ax.set_xlim(0, len(costs))
        ax.set_ylim(0, max(costs))
        return ln,

    def update(frame):
        x_data.append(frame)
        y_data.append(costs[frame])
        ln.set_data(x_data, y_data)
        return ln,

    ani = FuncAnimation(fig, update, frames=range(len(costs)),
                        init_func=init, blit=True)
    plt.show()

def sample_data(index, X, Y, classes):
    plt.imshow(X[index])
    plt.show()

if __name__ == "__main__":
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
    train_set_x, test_set_x = flatten_input_images(train_set_x_orig, test_set_x_orig)
    train_set_x = normalize_image_data(train_set_x)
    test_set_x = normalize_image_data(test_set_x)

    num_iterations = [1_000]
    learning_rates = [0.05]
    for learning_rate in learning_rates:
        for num_iteration in num_iterations:
            print("Testing with learning_rate={}, num_iterations={}".format(learning_rate, num_iteration))
            model_params, costs = nn_model(train_set_x, train_set_y, num_iteration, learning_rate, True)
            # plot_cost_function(costs)
            test_predictions = predict(test_set_x, model_params)
            train_predictions = predict(train_set_x, model_params)
            test_accuracy = 100 - np.mean(np.abs(test_predictions - test_set_y)) * 100
            train_accuracy = 100 - np.mean(np.abs(train_predictions - train_set_y)) * 100

            print("train accuracy: {}%".format(train_accuracy))
            print("test accuracy: {}%".format(test_accuracy))
            false_negatives = 0
            false_positives = 0
            m_test = test_set_y.shape[1]

            print("the model thought it wasn't a cat, but it was a cat!")
            for i, prediction in enumerate(test_predictions):
                if prediction == 0 and prediction != test_set_y[0][i]:
                    # sample_data(i, test_set_x_orig, test_set_y, classes)
                    false_negatives += 1

            print("the model thought it was a cat, but it wasn't!")
            for i, prediction in enumerate(test_predictions):
                if prediction == 1 and prediction != test_set_y[0][i]:
                    # sample_data(i, test_set_x_orig, test_set_y, classes)
                    false_positives += 1

            print("the model thought it was a cat, and it was!")
            for i, prediction in enumerate(test_predictions):
                if prediction == 1 and prediction == test_set_y[0][i]:
                    # sample_data(i, test_set_x_orig, test_set_y, classes)
                    pass

            print("the model thought it wasn't a cat, and it wasn't!")
            for i, prediction in enumerate(test_predictions):
                if prediction == 0 and prediction == test_set_y[0][i]:
                    # sample_data(i, test_set_x_orig, test_set_y, classes)
                    pass

            print("Percent False positive {}%".format((false_positives / m_test) * 100))
            print("Percent False negatives {}%".format((false_negatives / m_test) * 100))


