from math import log
import numpy as np
import matplotlib.pyplot as plt

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoidPrime(x): return x * (1-x)

def relu(x): return np.maximum(0, x)
def reluPrime(x): return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

#Some useful comversion functions
def ListtoVector(new_list):
    length = len(new_list)
    vec = np.arange(length, dtype=float)
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
        #weight_matrix = 2 * np.random.rand(new_list[c+1], new_list[c]) - 1
        #bias_matrix = 2 * np.random.rand(new_list[c+1],1) - 1
        weight_matrix = np.random.randn(new_list[c+1], new_list[c]) * np.sqrt(2 / new_list[c])
        bias_matrix = np.zeros((new_list[c+1], 1))
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
            # normalize pixel values
            in_vec = ListtoVector([int(p) / 255 for p in image])
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
    # save activations of each layer for backprop
    activations = [inp]
    for i in range(1, len(weights)):
        #inp = A_vec(weights[i] @ inp + biases[i])
        #activations.append(inp)
        z = weights[i] @ activations[-1] + biases[i]
        if i == len(weights) - 1:
            activations.append(softmax(z))    # last layer: softmax
        else:
            activations.append(A_vec(z))
    # activations[-1] is the output vector
    return activations


#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):

    # choose our learning rate
    learning_rate = 0.001
    
    #initialize se_sum and correct
    se_sum = 0
    correct = 0


    # loop through each training example
    for i in range(len(training)):
        # get activations
        activations = p_net(relu, weights, biases, training[i][0])

        # compute se_sum for current training example
        se_sum += np.sum((activations[-1] - training[i][1]) ** 2) / len(activations[-1])

        # check if prediction was correct for current training example (accuracy tracking)
        predicted_number = np.argmax(activations[-1])
        expected_number = np.argmax(training[i][1])
        if predicted_number == expected_number:
            correct += 1

        # compute delta for last layer
        #delta = sigmoidPrime(activations[-1]) * (activations[-1] - training[i][1])

        delta = activations[-1] - training[i][1]

        # loop through each layer of the network and update weights and biases
        for j in range(len(weights) - 1, 0, -1):
            # calculate new delta (needs to be calculated before the weight updates)
            #new_delta = (weights[j].T @ delta) * reluPrime(activations[j-1])
            
            if j > 1:
                # hidden layers use reluPrime
                new_delta = (weights[j].T @ delta) * reluPrime(activations[j-1])
            else:
                # j-1 == 0 is the input layer
                new_delta = weights[j].T @ delta

            # update weight
            weights[j] = weights[j] - learning_rate * (delta @ activations[j-1].T)

            # update biases
            biases[j] = biases[j] - learning_rate * (delta)

            # calculate new delta (for next layer based on current layer)
            delta = new_delta


    # calculate MSE and accuracy for epoch 
    mse = se_sum / len(training)
    accuracy = correct / len(training)

    # method returns the MSE for the epoch for tracking purposes
    return weights, biases, mse, accuracy

    
#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
def run_model(epochs, training_file, testing_file, hidden_1, hidden_2):
    
    # read and setup training and test data
    training_data = read_file(training_file)
    testing_data = read_file(testing_file)

    # setup weights and biases
    weights, biases = architecture([784, hidden_1, hidden_2, 10])


    # initialize accuracy lists
    training_accuracy_progression = []
    testing_accuracy_progression = []


    for i in range(1, epochs + 1):
        weights, biases, mse, training_accuracy = one_epoch(training_data, weights, biases)

        # calculate accuracy for testing data
        testing_accuracy = get_accuracy(testing_data, weights, biases)

        testing_accuracy_progression.append(testing_accuracy)
        training_accuracy_progression.append(training_accuracy)
        print("Epoch " + str(i) + " testing accuracy: " + str(round(testing_accuracy,4)))

    # plot accuracies over epochs
    plt.plot(range(1, epochs + 1), testing_accuracy_progression, label="Testing")
    plt.plot(range(1, epochs + 1), training_accuracy_progression, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy Error Over Epochs")
    plt.show()

    return testing_accuracy_progression

def get_accuracy(data, weights, biases):
    correct = 0
    for i in range(len(data)):
        output = p_net(relu, weights, biases, data[i][0])
        predicted = np.argmax(output[-1])    
        expected = np.argmax(data[i][1])
        if predicted == expected:
            correct += 1
    return correct / len(data)


run_model(20, 'mnist_train.csv', 'mnist_test.csv', 256, 128)




    






