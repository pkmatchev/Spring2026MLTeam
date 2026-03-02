from math import log
import sys
import ast
import numpy as np
import random
import pickle

def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


def p_net(A_vec, weights, biases, inp):
    for i,v in enumerate(weights):
        if i >= 1:
            inp = A_vec(v@inp + biases[i])
    return inp


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

def one_epoch(training, weights, biases):
    network_length = len(weights) - 1
    learning_rate = 0.1
    for v in training:
        inp = v[0]/255
        output = v[1]
        a = list()
        dots = list()
        deltas = list()
        dots.append(None)
        a.append(inp)
        for c in range(1,network_length+1):
            dots.append(weights[c]@a[c-1] + biases[c])
            a.append(sigmoid(dots[c]))
        for c in range(network_length+1):
            deltas.append(None)
        deltas[network_length] = sigmoidPrime(dots[network_length])*(a[network_length] - output)
        for c in range(network_length-1, 0, -1):
            deltas[c] = (sigmoidPrime(dots[c])*(weights[c+1].transpose()@deltas[c+1]))
        for c in range(1, network_length+1):
            biases[c] = biases[c] - learning_rate*deltas[c]
            weights[c] = weights[c] - learning_rate*(deltas[c]@(a[c-1].transpose()))
    return weights, biases

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

train_points = read_file("mnist_small_train.csv")
test_points = read_file("mnist_small_test.csv")
num_epochs = 0
arch = [784, 500, 150, 10] #Network architecture
w, b = architecture(arch)
while num_epochs < 200:
    num_correct = 0
    nW, nB = one_epoch(train_points, w, b)
    num_epochs += 1
    for v in train_points:
        count = 0
        inp = v[0]/255
        output = v[1]
        out_list = VectortoList(output)
        result = (p_net(sigmoid, nW, nB, inp))
        res_list = VectortoList(result)
        best_index = res_list.index(max(res_list))
        if out_list[best_index] == 1:
            num_correct += 1
    print("The architecture of the network is", arch,".")
    print("The percent accuracy in the training set after", num_epochs, "epochs is", float(100)*float(num_correct/len(train_points)), "percent.")
    num_correct = 0
    for v in test_points:
        count = 0
        inp = v[0]/255
        output = v[1]
        out_list = VectortoList(output)
        result = (p_net(sigmoid, nW, nB, inp))
        res_list = VectortoList(result)
        best_index = res_list.index(max(res_list))
        if out_list[best_index] == 1:
            num_correct += 1
    print("The percent accuracy in the test set after", num_epochs, "epochs is", float(100)*float(num_correct/len(test_points)), "percent.")





