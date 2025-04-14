import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import pickle
import sys

# Add evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator

class SklearnLinearSVM:
    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        #initialize the model
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LinearSVC(
            C=self.C,
            max_iter=self.max_iter,
            dual=False,  
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        
    def train(self, X, y, scale_data=True, verbose=False):

        # Scale data if requested
        if scale_data:
            X = self.scaler.fit_transform(X)
        
        if verbose:
            print("Training LinearSVC model...")
            
        # Train the model
        self.model.fit(X, y)
        
        if verbose:
            print("Training complete.")
            
        return self
    
    def predict(self, X):

        # Scale data if scaler exists
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        #calculate accuracy
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def decision_function(self, X):

        X = self.scaler.transform(X)
        return self.model.decision_function(X)
    
    def save_model(self, save_path: str):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Set training dataset path
    BASE_DIR = r"API\data\processed"
    train_path = os.path.join(BASE_DIR, "train_data.csv")
    test_path = os.path.join(BASE_DIR, "test_data.csv")

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    target_column = train_data.columns[-1]  

    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values

    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values

    # Create and train the model
    svm = SklearnLinearSVM(C=100, max_iter=1000, random_state=42)
    svm.train(X_train, y_train, scale_data=True, verbose=True)
    
    # Evaluate on test data
    test_accuracy = svm.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    y_pred = svm.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model
    model_save_path = r"..\API\model\saved_models\LinearSVM.pkl"
    svm.save_model(model_save_path)

