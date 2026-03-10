from math import log
import numpy as np
import matplotlib.pyplot as plt

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


#Some useful comversion functions
def ListtoVector(new_list):
    length = len(new_list)
    # Create a float array instead of integers
    vec = np.zeros(length, dtype=float)
    for i, v in enumerate(new_list):
        vec[i] = float(v)  # make sure we store floats
    return vec.reshape(length, 1)

def VectortoList(new_vec):
    length = (new_vec.size)
    reshaped = new_vec.reshape(1, length)
    new_list = list()
    for v in reshaped[0]:
        new_list.append(v)
    return new_list

#A function that setups your initial network including randomized weights, input is the form of a list of the number of neurons in each layer. e.g [784, x, y, 10] where x and y are the number of neurons in your hiddenn layers
def architecture(new_list):
    weights = list()
    biases = list()
    weights.append(None)
    biases.append(None)
    network_length = len(new_list)
    for c in range(network_length-1):
        weight_matrix = 2 * np.random.rand(new_list[c+1], new_list[c]) - 1
        bias_matrix = 2 * np.random.rand(new_list[c+1],1) - 1
        weights.append(weight_matrix)
        biases.append(bias_matrix)
    return weights, biases

#take in the CSVs and vectorize the output, would recommend experimenting with to see exactly what happens
def read_file(file_name):
    toReturn = list()
    with open(file_name) as f:
        for line in f:
            image = line[0:len(line)-1].split(",")
            output = image.pop(0)

            # normalize pixel values
            image = [float(p)/255 for p in image]

            in_vec = ListtoVector(image)
            out_vec = list()
            for c in range(10):
                if c == int(output):
                    out_vec.append(1)
                else:
                    out_vec.append(0)
            out_vec = ListtoVector(out_vec)
            toAppend = (in_vec,out_vec)
            toReturn.append(toAppend)
    return toReturn

#TODO A feed forward of the network where A_vec is the activation function, weights is a list of all the weight matrices, biases is a list of all the bias vectors, and inp is the input, return the output as a vector
def p_net(A_vec, weights, biases, inp):
    a = inp
    for i in range(1, len(weights)):
        z = weights[i] @ a + biases[i]
        a = A_vec(z)
    return a

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    learning_rate = 0.1

    for inp, target in training:
        activations = [inp]
        zs = [None]

        a = inp

        # Forward Pass
        for i in range(1, len(weights)):
            z = weights[i] @ a + biases[i]
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # Backprop
        delta = (activations[-1] - target) * sigmoidPrime(zs[-1])

        # Update last layer
        weights[-1] = weights[-1] - learning_rate * (delta @ activations[-2].T)
        biases[-1] = biases[-1] - learning_rate * delta

        # Backprop through Hidden Layers
        for l in range(len(weights) - 2, 0, -1):
            delta = (weights[l+1].T @ delta) * sigmoidPrime(zs[l])
            weights[l] = weights[l] - learning_rate * (delta @ activations[l-1].T)
            biases[l] = biases[l] - learning_rate * delta
    
    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
def accuracy(data, weights, biases):
    correct = 0

    for inp, target in data:
        output = p_net(sigmoid, weights, biases, inp)

        predicted = np.argmax(output)
        actual = np.argmax(target)

        if predicted == actual:
            correct += 1

    return correct / len(data)

def train_model(train_data, test_data, weights, biases, epochs=10):

    train_acc = []
    test_acc = []

    for e in range(epochs):

        weights, biases = one_epoch(train_data, weights, biases)

        train_accuracy = accuracy(train_data, weights, biases)
        test_accuracy = accuracy(test_data, weights, biases)

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        print("Epoch", e+1, 
              "Train Accuracy:", train_accuracy, 
              "Test Accuracy:", test_accuracy)

    plt.plot(range(1, epochs+1), train_acc, label="Train Accuracy")
    plt.plot(range(1, epochs+1), test_acc, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.show()

train_data = read_file("mnist_train.csv")
test_data = read_file("mnist_test.csv")

weights, biases = architecture([784, 100, 50, 10])

train_model(train_data, test_data, weights, biases, epochs=10)





