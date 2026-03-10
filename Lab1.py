from math import log
import numpy as np
import matplotlib.pyplot as plt

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
    num_layers = len(weights)
    z_list = z_list = [None]
    a_list = [inp]

    a = inp
    for i in range(1, num_layers):
        z = weights[i] @ a + biases[i]
        a = A_vec(z)
        z_list.append(z)
        a_list.append(a)
    return a, z_list, a_list

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases, learning_rate):
     num_layers = len(weights)
     for inp, target in training:
        output, z_list, a_list = p_net(sigmoid, weights, biases, inp)
        deltas = [None] * num_layers
        deltas[num_layers - 1] = (a_list[num_layers - 1] - target) * sigmoidPrime(z_list[num_layers - 1])

        for l in range(num_layers - 2, 0, -1):
            deltas[l] = (weights[l + 1].T @ deltas[l + 1]) * sigmoidPrime(z_list[l])
        
        for l in range(1, num_layers):
            weights[l] -= learning_rate * (deltas[l] @ a_list[l - 1].T)
            biases[l] -= learning_rate * deltas[l]
     return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
def accuracy(data, weights, biases):
    correct = 0
    for inp, target in data:
        output, x, y= p_net(sigmoid, weights, biases, inp)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    return correct / len(data)

training_data = read_file("mnist_train.csv")
test_data = read_file("mnist_test.csv")
layers = [1000, 300, 50, 10]
weights, biases = architecture(layers)
num_epochs = 15
lr = 0.2

train_accuracies = []
test_accuracies = []

for epoch in range(1, num_epochs + 1):
    np.random.shuffle(training_data)

    weights, biases = one_epoch(training_data, weights, biases, lr)

    train_acc = accuracy(training_data, weights, biases)
    test_acc = accuracy(test_data, weights, biases)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-o', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-o', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'MNIST Neural Network — Architecture {layers}')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.savefig('accuracy_plot.png', dpi=150)
plt.show()


