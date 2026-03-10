from math import log
import numpy as np

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


#Some useful comversion functions
def ListtoVector(new_list):
    length = len(new_list)
    vec = np.arange(length)
    for i,v in enumerate(new_list):
        vec[i] = v
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

#take in the CSVs and vectorize the output, would reccomend experimenting with to see exactly what happens
def read_file(file_name):
    toReturn = list()
    with open(file_name) as f:
        for line in f:
            image = line[0:len(line)-1].split(",")
            output = image.pop(0)
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
    output = inp
    for c in range(1, len(weights)):
        output = A_vec(weights[c] @ output + biases[c])
    return output

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    network_length = len(weights) - 1
    learning_rate = 0.1
    for v in training:
        inp = v[0] / 255
        output = v[1]/
        a = [inp]
        dots = [None]
        for c in range(1, network_length + 1):
            dots.append(weights[c] @ a[c-1] + biases[c])
            a.append(sigmoid(dots[c]))
        # Phase 2: Compute deltas backwards
        deltas = [None] * (network_length + 1)
        deltas[network_length] = sigmoidPrime(dots[network_length]) * (a[network_length] - output)
        for c in range(network_length - 1, 0, -1):
            deltas[c] = sigmoidPrime(dots[c]) * (weights[c+1].T @ deltas[c+1])
        # Phase 3: Update weights and biases
        for c in range(1, network_length + 1):
            biases[c] = biases[c] - learning_rate * deltas[c]
            weights[c] = weights[c] - learning_rate * (deltas[c] @ a[c-1].T)
    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch

train_points = read_file("mnist_small_train.csv")
test_points = read_file("mnist_small_test.csv")
epochs = 0
network_arch = [784, 400, 200, 10]
w, b = architecture(network_arch)

import matplotlib.pyplot as plt

train_accuracies = []
test_accuracies = []
num_epochs = 20

while epochs < num_epochs:
    w, b = one_epoch(train_points, w, b)
    epochs += 1

    # Train accuracy
    num_correct = 0
    for v in train_points:
        inp = v[0] / 255
        output = v[1]
        out_list = VectortoList(output)
        result = p_net(sigmoid, w, b, inp)
        res_list = VectortoList(result)
        best_index = res_list.index(max(res_list))
        if out_list[best_index] == 1:
            num_correct += 1
    train_acc = 100.0 * num_correct / len(train_points)
    train_accuracies.append(train_acc)

    # Test accuracy
    num_correct = 0
    for v in test_points:
        inp = v[0] / 255
        output = v[1]
        out_list = VectortoList(output)
        result = p_net(sigmoid, w, b, inp)
        res_list = VectortoList(result)
        best_index = res_list.index(max(res_list))
        if out_list[best_index] == 1:
            num_correct += 1
    test_acc = 100.0 * num_correct / len(test_points)
    test_accuracies.append(test_acc)

    print(f"Epoch {epochs}: Train = {train_acc:.2f}%, Test = {test_acc:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"Train vs Test Accuracy — Architecture: {network_arch}")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()
