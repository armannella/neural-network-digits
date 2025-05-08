# Handwritten Digit Recognition – My First Neural Network Project (Course Assignment) 🧠

This is my **first neural network project**, built entirely from scratch in Python as part of a **university course project for a Devices for AI class**.

It demonstrates a basic understanding of feedforward neural networks, without relying on machine learning libraries like TensorFlow or PyTorch. The model is designed to recognize **handwritten digits** from image data.

## Features

- Manual implementation of `ReLU`, `Softmax`, and `Cross Entropy Loss`
- Full forward pass and backpropagation logic
- Reads training and testing data from CSV files (MNIST-style)
- Prints training loss per epoch and final classification accuracy

## Network Architecture

```python
layers_size = [784, 16, 16, 10]
```

- 784 input neurons (28x28 pixel grayscale images)
- Two hidden layers with 16 neurons each
- 10 output neurons for digit classification (0–9)

## How to Run

1. Place `train.csv` and `test.csv` files in the same folder as `neuron_network.py`.
2. Run the script with:

```bash
python neuron_network.py
```

3. The script will display training loss and final accuracy.

## Requirements

- Python 3.x
- No external libraries required


## License

This project is released under the MIT License.