import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import pickle

# Add evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator


class SklearnRBFSVM:
    def __init__(self, C=100.0, gamma=0.001, random_state=42,kernel ='rbf'):
        #initialize the model
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(
            kernel=kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state
        )
        
    def train(self, X, y, verbose=False):

        if verbose:
            print(f"Training sklearn's SVC with RBF kernel (C={self.C}, gamma={self.gamma})...")
            
        # Train the model
        self.model.fit(X, y)
        
        if verbose:
            print("Training complete.")
            
        return self
    
    def predict(self, X):
        #give predictions
        return self.model.predict(X)
    
    def score(self, X, y):
        #calculate accuracy score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def decision_function(self, X):

        return self.model.decision_function(X)
    
    #save model
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)
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
    rbf_svm = SklearnRBFSVM(C=10.0, gamma=0.01, random_state=42,kernel ='rbf')
    rbf_svm.train(X_train, y_train, verbose=True)
    
    # Evaluate on test data
    test_accuracy = rbf_svm.score(X_test, y_test)
    print(f"Sklearn SVC with RBF Kernel Accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    y_pred = rbf_svm.predict(X_test)
    print("\nClassification Report (Sklearn SVC with RBF Kernel):")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model
    model_save_path = r"API\model\saved_models\Non_LinearSVC.pkl"
    rbf_svm.save_model(model_save_path)
