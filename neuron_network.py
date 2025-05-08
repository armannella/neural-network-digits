import random
import math
import csv

# ***************************************** Functions ***************************************** 

# ---------- Part 1 (Initialization) Functions -------- :

def initialize_weights(layers_size):
    weights = []
    for k in range(len(layers_size) - 1):
        layer_weights = []
        for i in range(layers_size[k + 1]):
            row = [random.uniform(-1, 1) * math.sqrt(2 / layers_size[k]) for _ in range(layers_size[k])]
            layer_weights.append(row)
        weights.append(layer_weights)
    return weights

def initialize_biases(layers_size):
    biases = []
    for k in range(1, len(layers_size)):
        biases.append([0.0 for _ in range(layers_size[k])])
    return biases


# # ---------- Part 2 & 3 (Training & Testing) Functions -------- :


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def forward_pass(layers_size, values, weights, biases):
    for layer in range(1, len(layers_size)):
        layer_values = []
        for neuron in range(layers_size[layer]):
            z = sum(values[layer - 1][i] * weights[layer - 1][neuron][i] for i in range(layers_size[layer - 1])) + biases[layer - 1][neuron]
            if layer == len(layers_size) - 1:  
                layer_values.append(z)
            else:
                layer_values.append(relu(z))
        if layer == len(layers_size) - 1:
            values.append(softmax(layer_values))
        else:
            values.append(layer_values)
    return values

def backward_pass(layers_size, values, weights, biases, labels, learning_rate):
    gradients = [None] * len(layers_size)

    gradients[-1] = [(values[-1][i] - labels[i]) for i in range(layers_size[-1])]

    
    for layer in range(len(layers_size) - 2, 0, -1):
        gradients[layer] = []
        for neuron in range(layers_size[layer]):
            gradient = sum(
                gradients[layer + 1][i] * weights[layer][i][neuron] * relu_derivative(values[layer][neuron])
                for i in range(layers_size[layer + 1])
            )
            gradients[layer].append(gradient)

    
    for layer in range(len(weights)):
        for i in range(len(weights[layer])):
            for j in range(len(weights[layer][i])):
                weights[layer][i][j] -= learning_rate * gradients[layer + 1][i] * values[layer][j]
        for i in range(len(biases[layer])):
            biases[layer][i] -= learning_rate * gradients[layer + 1][i]

def cross_entropy_loss(y_true, y_pred):
    return -sum(y_true[i] * math.log(y_pred[i]) for i in range(len(y_true)))




# ***************************************** Main Code ***************************************** 


# -------------------- Part 1 (Initialization)  -------------------- :

layers_size = [784, 16, 16, 10]
weights = initialize_weights(layers_size)
biases = initialize_biases(layers_size)
learning_rate = 0.01



# -------------------- Part 2 (Training) -------------------- :

file_path = 'train.csv'
epochs = 10

for epoch in range(epochs):
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        total_loss = 0
        for row in csv_reader:
            label = int(row[0])
            input_label = [0] * 10
            input_label[label] = 1

            # 2.1 : Normalization Input Bits :
            values = [[float(value) / 255 for value in row[1:]]]

            # 2.2 : Forward Pass :
            values = forward_pass(layers_size, values, weights, biases)

            # 2.3 : Calculate Cost Function :
            total_loss += cross_entropy_loss(input_label, values[-1])

            # 2.4 : Backpropogation            
            backward_pass(layers_size, values, weights, biases, input_label, learning_rate)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")




# -------------------- Part 3 (Testing) -------------------- :


file_path = 'test.csv'
correct = 0
sample = 0

with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        label = int(row[0])

        # 3.1 : Normalization Input Bits :
        values = [[float(value) / 255 for value in row[1:]]]

        # 3.2 : Forward Pass :
        values = forward_pass(layers_size, values, weights, biases)

        # 3.3 : Check Answer :
        prediction = values[-1].index(max(values[-1]))
        if prediction == label:
            correct += 1
        sample += 1

print(f"Accuracy: {correct / sample * 100:.2f}%")