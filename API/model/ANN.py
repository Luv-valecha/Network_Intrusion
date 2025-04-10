import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os, sys

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 



# Part 1: Function Implementations

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):

    sig=sigmoid(z)
    return sig*(1-sig)
    

def relu(z):

    return np.maximum(0, z)
    

def relu_derivative(z):

    return np.where(z > 0, 1, 0)
    

def linear(z):

    return z
    

def linear_derivative(z):

    return np.ones_like(z)
    

def mean_squared_error(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)
    

# Part 2: NeuralNetwork Class Implementation

class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'):

       self.layers = layers
       self.activation = activation
       self.weights = []
       self.biases = []

       # Initialize weights and biases randomly
       for i in range(len(layers) - 1):
           self.weights.append(np.random.randn(layers[i], layers[i + 1]))
           self.biases.append(np.random.randn(layers[i + 1]))

    def forward(self, x):
        activations = [x]
        z_values = []
        # run for every layer
        for i in range(len(self.layers) - 1):
            # z value for this layer
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)

            # applying activation function
            if self.activation == 'sigmoid':
                a = sigmoid(z)
            elif self.activation == 'relu':
                a = relu(z)
            elif self.activation == 'linear':
                a = linear(z)
            activations.append(a)
        return activations, z_values
        

    def backward(self, x, y, activations, z_values, learning_rate):
        m = y.shape[0]  # Number of samples
        delta = activations[-1] - y  # Output layer error

        if self.activation == 'sigmoid':
            delta *= sigmoid_derivative(z_values[-1])
        elif self.activation == 'relu':
            delta *= relu_derivative(z_values[-1])
        elif self.activation == 'linear':
            delta *= linear_derivative(z_values[-1])

        # Loop backwards through the layers
        for i in range(len(self.layers) - 2, -1, -1):
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                if self.activation == 'sigmoid':
                    delta *= sigmoid_derivative(z_values[i - 1])
                elif self.activation == 'relu':
                    delta *= relu_derivative(z_values[i - 1])
                elif self.activation == 'linear':
                    delta *= linear_derivative(z_values[i - 1])


    def train(self, x_train, y_train, epochs, learning_rate, x_val=None, y_val=None):
        y_train = y_train.reshape(-1, 1)
        if x_val is not None:
            y_val = y_val.reshape(-1, 1)

        for epoch in range(epochs):
            activations, z_values = self.forward(x_train)
            loss = mean_squared_error(y_train, activations[-1])

            if epoch % 50 == 0:
                val_loss = 0
                if x_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = mean_squared_error(y_val, val_pred)
                    print(f'Epoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch}, Training Loss: {loss:.4f}')

            self.backward(x_train, y_train, activations, z_values, learning_rate)


    def predict(self, x_test):
        activations, _ = self.forward(x_test)
        probabilities = activations[-1]
        return (probabilities >= 0.5).astype(int)

    def save_model(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {filename}")
        except (OSError, IOError) as e:
            print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"..\data\processed\train_data.csv" 
    test_dataset_path = r"..\data\processed\test_data.csv"

    # Read the datapaths
    train = pd.read_csv(train_dataset_path)
    train_label = train["class"]
    train.drop('class',inplace=True,axis=1)

    test = pd.read_csv(test_dataset_path)
    test_label = test["class"]
    test.drop('class',inplace=True,axis=1)

    # Split training into actual training and validation
    x_train, x_val, y_train, y_val = train_test_split(train, train_label, test_size=0.2, random_state=42)

    # Initialize classifier with training data
    classifier = NeuralNetwork([10,128, 64, 32, 16, 1], activation='sigmoid')
    classifier.train(
        np.array(x_train), np.array(y_train),
        epochs=1500,
        learning_rate=0.5,
        x_val=np.array(x_val),
        y_val=np.array(y_val)
    )

    # Save the trained model
    model_save_path = r"..\model\saved_models\ann.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("ann.pkl")
    evaluater.evaluate()