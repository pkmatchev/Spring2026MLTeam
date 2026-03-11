from math import log
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return sigmoid(x) * (1 - sigmoid(x))


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
            in_vec = ListtoVector(image) / 255.0
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
def p_net(A_vec, weights, biases, input):

    # Store activations and zs
    activations = [input]
    zs = []

    # Iterate through layers and append each new activation
    for current_weight, current_bias in zip(weights, biases):
        z = current_weight @ activations[-1] + current_bias
        zs.append(z)
        activations.append(A_vec(z))

    return activations, zs

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):

    # Define learning rate
    learning_rate = 0.1

    # Iterate over training data
    for input, target in training:

        # Forward prop
        activations, zs = p_net(sigmoid, weights, biases, input)

        # Calcuate the output error
        error = (activations[-1] - target) * sigmoidPrime(zs[-1])

        # Store the weight
        next_weight = weights[-1].copy()

        # Update weights and biases for the final layer
        weights[-1] -= learning_rate * (error @ activations[-2].T)
        biases[-1] -= learning_rate * error

        # Iterate backwards through layers to calculate back prop and adjust weights and biases
        for i in range(len(weights) - 2, -1, -1):

            # Calculate loss for the current layer
            error = (next_weight.T @ error) * sigmoidPrime(zs[i])

            # Store the weight
            next_weight = weights[i].copy()

            # Updates weights and biases
            weights[i] -= learning_rate * (error @ activations[i].T)
            biases[i] -= learning_rate * error

    return weights, biases

# Test the accuracy of the current weight and biases
def evaluate_accuracy(data, weights, biases):

    # Store counter
    counter = 0

    # Iterate through data
    for input, target in data:

        # Find output prediction
        prediction = np.argmax(p_net(sigmoid, weights, biases, input)[0][-1])
        actual = np.argmax(target)
        
        # If the prediction is correct increase accuracy
        if prediction == actual:
            counter += 1

    return counter / len(data)

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch

if __name__ == '__main__':
    # Settings
    layers = [784, 20, 20, 10]
    epochs = 10
    
    # Read in data
    training = read_file('mnist_train.csv')
    testing = read_file('mnist_test.csv')

    # Set up architecture
    train_weights, train_biases = architecture(layers)
    train_weights = train_weights[1:]
    train_biases = train_biases[1:]
    
    # Book keeping
    train_accuracies = []
    test_accuracies = []

    # Iterate across epochs
    for epoch in range(epochs):
        # Train on train data
        train_weights, train_biases = one_epoch(training, train_weights, train_biases)

        # Evaluate train accuracy
        train_accuracy = evaluate_accuracy(training, train_weights, train_biases)
        train_accuracies.append({
            'Epoch': epoch,
            'Accuracy': train_accuracy
        })

        # Evaluate test accuracy
        test_accuracy = evaluate_accuracy(testing, train_weights, train_biases)
        test_accuracies.append({
            'Epoch': epoch,
            'Accuracy': test_accuracy
        })

    # Convert to DataFrame for ease
    train_accuracies_df = pd.DataFrame(train_accuracies)
    test_accuracies_df = pd.DataFrame(test_accuracies)

    # Plot
    sns.lineplot(data = train_accuracies_df, x = 'Epoch', y = 'Accuracy', label = 'Train')
    sns.lineplot(data = test_accuracies_df, x = 'Epoch', y = 'Accuracy', label = 'Test')
    plt.title('Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()







