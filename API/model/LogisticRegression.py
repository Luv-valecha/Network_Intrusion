import numpy as np
import pandas as pd
import pickle
import json
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    """
    Wrapper for sklearn LogisticRegression to match custom class structure.
    Loads hyperparameters from external JSON file.
    """
    def __init__(self):
        # Load best hyperparameters from file
        hparam_path = r"API\model\Hyperparams\logistic_regression_hparam.json"
        with open(hparam_path, "r") as f:
            params = json.load(f)

        self.model = SklearnLogisticRegression(solver='lbfgs', **params)
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y_true):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)
        return accuracy, report, matrix

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"

    # Read the datasets
    train = pd.read_csv(train_dataset_path)
    test = pd.read_csv(test_dataset_path)

    # Separate features and labels
    train_label = train["class"].values
    test_label = test["class"].values
    train.drop('class', inplace=True, axis=1)
    test.drop('class', inplace=True, axis=1)

    # Initialize classifier using best hyperparameters
    classifier = LogisticRegression()
    classifier.train(train, train_label)

    # Save the trained model
    model_save_path = r"API\model\saved_models\logistic_regression.pkl"
    classifier.save_model(model_save_path)
