import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import pickle, json

# LinearSVM Implementation
class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def _initialize_weights(self, n_features):
        #Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
    def _hinge_loss(self, X, y):
        
       # Calculate the hinge loss and gradients
        
        n_samples = X.shape[0]
        
        # Calculate linear scores
        scores = np.dot(X, self.w) + self.b
        
        # Calculate margins (y_i * (w^T * x_i + b))
        margins = y * scores
        
        # Calculate hinge loss components
        hinge_loss_components = np.maximum(0, 1 - margins)
        
        # Total regularized loss
        loss = np.sum(hinge_loss_components) / n_samples + (self.lambda_param / 2) * np.dot(self.w, self.w)
        
        # Gradient calculation
        mask = (hinge_loss_components > 0).astype(int)
        dw = self.lambda_param * self.w - np.sum(X * (y * mask).reshape(-1, 1), axis=0) / n_samples
        db = -np.sum(y * mask) / n_samples
        
        return loss, dw, db
    
    def train(self, X, y, verbose=False):
        
        #Train the SVM model using gradient descent
        
  
        # Convert y to -1, 1 format 
        y_binary = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        loss_history = []
        
        iterator = tqdm(range(self.n_iterations)) if verbose else range(self.n_iterations)
        
        for i in iterator:
            # Calculate loss and gradients
            loss, dw, db = self._hinge_loss(X, y_binary)
            loss_history.append(loss)
            
            # Update weights and bias using gradient descent
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # Print progress
            if verbose and (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.6f}")
                
        return loss_history
    
    def predict(self, X):
        #Predict class labels for samples in X
 
        scores = np.dot(X, self.w) + self.b
        y_pred = np.where(scores < 0, 0, 1)  # Convert back to 0, 1 format
        return y_pred
    
    def score(self, X, y):
        #Calculate accuracy
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def decision_function(self, X):
       # Calculate decision function values (distance from the hyperplane)

        return np.dot(X, self.w) + self.b
    
    def save_model(self, save_path: str):
        #Save the trained model to a file.
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # check if  the directory exists
        with open(save_path,"wb") as f:
            pickle.dump(self.model,f)
            print(" Model saved")

if __name__ == "__main__":
    # Set training dataset path
    BASE_DIR = r"..\API\data\processed"
    train_path = os.path.join(BASE_DIR, "train_data.csv")
    test_path = os.path.join(BASE_DIR, "test_data.csv")

    # Initialize classifier with training data

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    target_column = train_data.columns[-1]  

    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values

    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iterations=1000)

    # Save the trained model
    model_save_path = r"API\model\saved_models\LinearSVM.pkl"
    LinearSVM.save_model(model_save_path)
