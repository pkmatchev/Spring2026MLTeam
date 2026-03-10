import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)

def list_to_vector(new_list):
    return np.array(new_list, dtype=float).reshape(-1, 1)

def vector_to_list(new_vec):
    return new_vec.flatten().tolist()

def build_architecture(new_list):
    weights = [None]
    biases = [None]
    for i in range(len(new_list)-1):
        w = 2 * np.random.rand(new_list[i+1], new_list[i]) - 1
        b = 2 * np.random.rand(new_list[i+1], 1) - 1
        weights.append(w)
        biases.append(b)
    return weights, biases

def read_file(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f.readlines():
            data = line.strip().split(",")
            label = int(data.pop(0))
            
            pixel_values = [float(v) / 255.0 for v in data]
            input_vector = list_to_vector(pixel_values)
            
            target_vector = np.zeros((10, 1))
            target_vector[label] = 1.0
            
            dataset.append((input_vector, target_vector))
    return dataset

def forward_pass(activation_func, weights, biases, inputs):
    activation = inputs
    for i in range(1, len(weights)):
        z = weights[i] @ activation + biases[i]
        activation = activation_func(z)
    return activation

def train_one_epoch(training_data, weights, biases, learning_rate=0.1):
    for inputs, expected_output in training_data:
        activations = [inputs]
        z_values = [None]
        current_activation = inputs
        
        for i in range(1, len(weights)):
            z = weights[i] @ current_activation + biases[i]
            z_values.append(z)
            current_activation = sigmoid(z)
            activations.append(current_activation)
        
        deltas = [None] * len(weights)
        
        output_error = activations[-1] - expected_output
        deltas[-1] = sigmoid_prime(z_values[-1]) * output_error
        
        for i in range(len(weights) - 2, 0, -1):
            deltas[i] = sigmoid_prime(z_values[i]) * (weights[i+1].T @ deltas[i+1])
        
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * (deltas[i] @ activations[i-1].T)
            biases[i] -= learning_rate * deltas[i]
            
    return weights, biases

def get_accuracy(dataset, weights, biases):
    correct_predictions = 0
    for inputs, expected_output in dataset:
        prediction = forward_pass(sigmoid, weights, biases, inputs)
        if np.argmax(prediction) == np.argmax(expected_output):
            correct_predictions += 1
    return correct_predictions / len(dataset)



if __name__ == "__main__":
    print("Loading data...")
    train_data = read_file("mnist_train.csv")
    test_data = read_file("mnist_test.csv")
    
    network_architecture = [784, 100, 10]
    w, b = build_architecture(network_architecture)
    
    train_accuracy_history = []
    test_accuracy_history = []
    total_epochs = 10
    
    print("Starting training...")
    for epoch in range(total_epochs):
        w, b = train_one_epoch(train_data, w, b, learning_rate=0.1)
        
        train_acc = get_accuracy(train_data, w, b)
        test_acc = get_accuracy(test_data, w, b)
        
        train_accuracy_history.append(train_acc)
        test_accuracy_history.append(test_acc)
        
        print(f"Epoch {epoch+1}/{total_epochs}: Train Acc = {train_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, total_epochs + 1), train_accuracy_history, label='Train Accuracy', marker='o')
    plt.plot(range(1, total_epochs + 1), test_accuracy_history, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()