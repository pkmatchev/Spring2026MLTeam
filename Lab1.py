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
    act = inp / 255

    for i in range(1, len(weights)):
        z = np.dot(weights[i], act) + biases[i]
        act = A_vec(z)
    
    return act

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    learning_rate = 0.1
    
    for inputs, actual in training:
        activations = [inputs / 255] 
        zvalues = []
        for i in range(1, len(weights)):
            z = np.dot(weights[i], activations[-1]) + biases[i]
            zvalues.append(z)
            activations.append(sigmoid(z))

        deltas = {}
        L = len(weights) - 1
        deltas[L] = (activations[-1] - actual) * sigmoidPrime(zvalues[-1])

        for l in range(L - 1, 0, -1):
            deltas[l] = np.dot(weights[l+1].T, deltas[l+1]) * sigmoidPrime(zvalues[l-1])

        for l in range(1, len(weights)):
            weights[l] = weights[l] - learning_rate * np.dot(deltas[l], activations[l-1].T)
            biases[l] = biases[l] - learning_rate * deltas[l]
    
    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
def train_and_evaluate(training_data, test_data, weights, biases, epochs = 10):
    train_accs = []
    test_accs = []

    for e in range(epochs):
        weights, biases = one_epoch(training_data, weights, biases)
        
        train_correct = sum(int(np.argmax(p_net(sigmoid, weights, biases, x)) == np.argmax(y)) for x, y in training_data)
        train_accs.append(train_correct / len(training_data))

        test_correct = sum(int(np.argmax(p_net(sigmoid, weights, biases, x)) == np.argmax(y)) for x, y in test_data)
        test_accs.append(test_correct / len(test_data))
        
        print(f"Epoch {e+1} | Train: {train_accs[-1]:.2%} | Test: {test_accs[-1]:.2%}")
        
    plt.plot(train_accs, label="Training Accuracy", color='blue')
    plt.plot(test_accs, label="Testing Accuracy", color='red')
    plt.legend()
    plt.show()

def main():
    print("Loading data...")
    training_data = read_file("mnist_train.csv")
    test_data = read_file("mnist_test.csv")

    layers = [784, 200, 100, 10]
    weights, biases = architecture(layers)

    print("Starting training...")
    train_and_evaluate(training_data, test_data, weights, biases, epochs = 20)

if __name__ == "__main__":
    main()