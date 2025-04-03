import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

class LogisticRegression:
    """
    Logistic Regression model implemented from scratch.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize the model with learning rate and number of epochs.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y):
        """
        Train the model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Predict class labels for input data.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]
    
    def save_model(self, save_path: str):
        """
        Save the trained model to a file.
        """
        with open(save_path, "wb") as f:
            pickle.dump({"weights": self.weights, "bias": self.bias}, f)
        print("Model saved successfully.")
    
    def load_model(self, load_path: str):
        """
        Load the model from a file.
        """
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
            self.weights = model_data["weights"]
            self.bias = model_data["bias"]
        print("Model loaded successfully.")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"

    # Read the datasets
    train = pd.read_csv(train_dataset_path)
    train_label = train["class"].values  # Convert to NumPy array
    train.drop('class', inplace=True, axis=1)
    train = train.values  # Convert to NumPy array

    test = pd.read_csv(test_dataset_path)
    test_label = test["class"].values  # Convert to NumPy array
    test.drop('class', inplace=True, axis=1)
    test = test.values  # Convert to NumPy array

    # Initialize classifier with training data
    classifier = LogisticRegression()
    classifier.train(train, train_label)

    # Save the trained model automatically after training
    model_save_path = r"API\model\saved_models\logistic_regression.pkl"
    classifier.save_model(model_save_path)  # Save the model

    print("Model training and saving completed.")
